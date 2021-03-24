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
        self.prev_actions = [None] * self.args.numAgents
        self.cuda_num = rank % args.n_gpus

    def oppo_agent(self, obs, config, prev_actions=None):
        prev_actions = self.prev_actions
        player = obs.index

        board = get_board(obs, prev_actions, self.args)
        board = get_player_board(board, player, self.args.numAgents)

        pi, _ = self.pnet.predict(board)

        # remove invalid action
        prev_action = prev_actions[player]
        if prev_action is not None:
            prev_action = str_to_action(prev_action)
            oppo_action = Action.opposite(prev_action).value
            pi[oppo_action - 1] = 0
            pi /= pi.sum()

        action = np.argmax(pi) + 1
        action = Action(action).name
        prev_actions[player] = action
        return action

    def playGame(self, player, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        env = make(
            "hungry_geese",
            configuration={
                "rows": self.args.boardSize[0],
                "columns": self.args.boardSize[1]
            },
            debug=False
        )
        agents = [agent if i == player else self.oppo_agent for i in range(self.args.numAgents)]
        agents[player].net = self.nnet
        env.run(agents)

        reward = get_reward(env.state[0].observation, player, self.args.numAgents)
        length = env.state[0].observation.step
        return reward, length


def playNGames(rank, arena, num, num_agents):
    player = rank % num_agents
    reward, length = 0, 0
    for _ in tqdm(range(num), desc=f"Arena.playGames ({player})"):
        r, l = arena.playGame(player)
        reward += r
        length += l
    return reward, length


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

    with mp.Pool(args.numProcesses) as p:
        params = [(rank, Arena(pnet, nnet, args, rank), num, args.numAgents) for rank in range(args.numProcesses)]
        rls = p.starmap(playNGames, params)

    for r, l in rls:
        reward += r
        length += l

    tot_num = num * args.numProcesses
    reward /= tot_num
    length /= tot_num
    return reward, length
