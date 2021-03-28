import logging
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

from agent import *

log = logging.getLogger(__name__)


class Arena():
    def __init__(self, pnet, nnet, args, rank):
        self.pnet = pnet
        self.nnet = nnet
        self.args = args
        self.rank = rank

    def playGame(self, player, verbose=False):
        """
        Executes one episode of a game.
        """
        env = make(
            "hungry_geese",
            configuration={
                "rows": self.args.boardSize[0],
                "columns": self.args.boardSize[1]
            },
            debug=False
        )
        env.reset(self.args.numAgents)

        prev_actions = [None] * self.args.numAgents

        while env.state[player]['status'] == 'ACTIVE' and not env.done:
            board = get_board(env.state[0].observation, prev_actions, self.args)

            # predict players' action
            actions = []
            pis, _ = self.pnet.predicts(board, self.rank % self.args.n_gpus)
            for i, pi in enumerate(pis):
                # this player uses nnet
                if i == player:
                    pi, _ = self.nnet.predict(board, player, self.rank % self.args.n_gpus)

                action = select_action(pi, prev_actions[i])
                actions.append(action)

            env.step(actions)

            prev_actions = actions

        reward = get_reward(env.state[0].observation, player, self.args.numAgents)
        length = env.state[0].observation.step
        return reward, length


def playNGames(rank, arena, num, num_agents, rls, q):
    player = rank % num_agents
    reward, length = 0, 0
    for _ in range(num):
        r, l = arena.playGame(player)
        reward += r
        length += l
        q.put(1)
    rls.append((reward, length))


def tqdm_listener(total, q):
    pbar = tqdm(total=total, desc="Arena.playGames")
    for _ in iter(q.get, None):
        pbar.update()


def playGames(pnet, nnet, num, args, verbose=False):
    """
    Plays num games in which player1 starts num/2 games and player2 starts
    num/2 games.

    Returns:
        oneWon: games won by player1
        twoWon: games won by player2
        draws:  games won by nobody
    """
    reward = 0
    length = 0

    with mp.Manager() as manager:
        rls = manager.list()
        q = mp.Queue()

        total_games = args.numProcesses * num
        tqdm_proc = mp.Process(target=tqdm_listener, args=(total_games, q))
        tqdm_proc.start()

        simulators = [
            mp.Process(
                target=playNGames,
                args=(
                    rank,
                    Arena(pnet, nnet, args, rank),
                    num,
                    args.numAgents,
                    rls,
                    q
                )
            )
            for rank in range(args.numProcesses)
        ]
        for simulator in simulators:
            simulator.start()
        for simulator in simulators:
            simulator.join()

        q.put(None)
        tqdm_proc.join()

        rls = list(rls)

    for r, l in rls:
        reward += r
        length += l

    tot_num = num * args.numProcesses
    reward /= tot_num
    length /= tot_num
    return reward, length
