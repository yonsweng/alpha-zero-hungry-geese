import logging
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
from kaggle_environments import make
from agent import *

log = logging.getLogger(__name__)


def playGame(pnet, nnet, args, player):
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
        pis, _ = pnet.predicts(board, player % args.n_gpus)
        for i, pi in enumerate(pis):
            # this player uses nnet
            if i == player:
                pi, _ = nnet.predict(board, player, player % args.n_gpus)

            action = select_action(pi, prev_actions[i])
            actions.append(action)

        env.step(actions)

        prev_actions = actions

    reward = get_reward(env.state[0].observation, player, args.numAgents)
    length = env.state[0].observation.step
    return reward, length

def playGames(pnet, nnet, args):
    """
    Returns:
        average_reward, average_length
    """

    with mp.Pool(processes=args.numProcesses) as p:
        func = partial(playGame, pnet, nnet, args)
        players = list(np.arange(args.arenaCompare) % args.numAgents)
        results = list(tqdm(p.imap(func, players), total=args.arenaCompare))

    total_reward = 0
    total_length = 0

    for reward, length in results:
        total_reward += reward
        total_length += length

    return total_reward / args.arenaCompare, total_length / args.arenaCompare
