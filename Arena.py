import logging
from tqdm import tqdm
import torch.multiprocessing as mp
from kaggle_environments import make
from agent import *

log = logging.getLogger(__name__)


def playGame(rank, pnet, nnet, player, args):
    """
    Executes one episode of a game.
    """
    env = make(
        "hungry_geese",
        configuration={
            "rows": args.boardSize[0],
            "columns": args.boardSize[1]
        },
        debug=False
    )
    env.reset(args.numAgents)

    prev_actions = [None] * args.numAgents

    while env.state[player]['status'] == 'ACTIVE' and not env.done:
        board = get_board(env.state[0].observation, prev_actions, args)

        # predict players' action
        actions = []
        pis, _ = pnet.predicts(board, rank % args.n_gpus)
        for i, pi in enumerate(pis):
            # this player uses nnet
            if i == player:
                pi, _ = nnet.predict(board, player, rank % args.n_gpus)

            action = select_action(pi, prev_actions[i])
            actions.append(action)

        env.step(actions)

        prev_actions = actions

    reward = get_reward(env.state[0].observation, player, args.numAgents)
    length = env.state[0].observation.step
    return reward, length


def playNGames(rank, pnet, nnet, num, args, rls, q):
    player = rank % args.numAgents
    reward, length = 0, 0
    for _ in range(num):
        r, l = playGame(rank, pnet, nnet, player, args)
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

        total_games = args.numProcessesArena * num
        tqdm_proc = mp.Process(target=tqdm_listener, args=(total_games, q))

        simulators = [
            mp.Process(
                target=playNGames,
                args=(
                    rank,
                    pnet,
                    nnet,
                    num,
                    args,
                    rls,
                    q
                )
            )
            for rank in range(args.numProcessesArena)
        ]

        tqdm_proc.start()
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

    tot_num = num * args.numProcessesArena
    reward /= tot_num
    length /= tot_num
    return reward, length
