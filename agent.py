import numpy as np
from itertools import permutations
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, adjacent_positions
from utils import args


def str_to_action(s):
    if s == 'NORTH':
        return Action.NORTH
    elif s == 'EAST':
        return Action.EAST
    elif s == 'SOUTH':
        return Action.SOUTH
    else:
        return Action.WEST


def get_board(obs, prev_actions, args):
    '''
    Channels of state:
        geese curr_heads  0,  1,  2,  3
        geese bodies      4,  5,  6,  7
        geese tips        8,  9, 10, 11
        geese prev_heads 12, 13, 14, 15
        food             16
    '''
    board = np.zeros((args.numAgents * 4 + 1, args.boardSize[0] * args.boardSize[1]), np.uint8)

    for i, goose in enumerate(obs.geese):
        # head position
        for head_pos in goose[:1]:
            board[0 + (i - obs.index) % args.numAgents, head_pos] = 1

        # tip position
        for tip_pos in goose[-1:]:
            board[args.numAgents + (i - obs.index) % args.numAgents, tip_pos] = 1

        # whole position
        for body_pos in goose[1:]:
            board[args.numAgents * 2 + (i - obs.index) % args.numAgents, body_pos] = 1

        # previous head position
        for head_pos in goose[:1]:
            if prev_actions[i] is not None:
                prev_action = str_to_action(prev_actions[i])
                opposite_action = Action.opposite(prev_action)
                prev_head_pos = adjacent_positions(head_pos, args.boardSize[1], args.boardSize[0])[opposite_action.value - 1]
                board[args.numAgents * 3 + (i - obs.index) % args.numAgents, prev_head_pos] = 1

    for food_pos in obs.food:
        board[-1, food_pos] = 1

    return board.reshape(-1, args.boardSize[0], args.boardSize[1])


def get_player_board(board, player, num_agents):
    new_board = board.copy()
    indices = np.arange(0, num_agents * 4, num_agents)
    tmp = new_board[indices]
    new_board[indices] = new_board[indices + player]
    new_board[indices + player] = tmp
    return new_board


def get_symmetries(board, player, num_agents, pi):
    boards = [(board, pi)]

    # 180 degree turn
    b = np.rot90(board, 2, (1, 2))
    p = [*pi[2:], *pi[:2]]
    boards.append((b, p))

    # flip ud
    b = np.flip(board, 1)
    p = pi[:]
    tmp = p[0]
    p[0] = p[2]
    p[2] = tmp
    boards.append((b, p))

    # flip lr
    b = np.flip(board, 2)
    p = pi[:]
    tmp = p[1]
    p[1] = p[3]
    p[3] = tmp
    boards.append((b, p))

    # change player indices
    indices = np.arange(0, num_agents * 4, num_agents)
    others = [i for i in range(num_agents) if i != player]
    for bb, p in boards[:4]:
        for pm in permutations(others, 3):
            b = bb.copy()
            for i, j in zip(others, pm):
                b[indices + i] = bb[indices + j]
            boards.append((b, p))

    return boards


def get_reward(obs, player, num_agents):
    alive = 0
    for goose in obs.geese:
        alive += 1 if len(goose) > 0 else 0

    if len(obs.geese[player]) > 0 and alive >= 2 and obs.step < 199:
        return None

    rank = 1
    for i, goose in enumerate(obs.geese):
        if i == player:
            continue
        if len(goose) > len(obs.geese[player]):
            rank += 1
        elif len(goose) == len(obs.geese[player]):
            rank += 0.5
    return (num_agents + 1 - 2 * rank) / (num_agents - 1)


def get_valid_moves(state, player, action_size):
    moves = np.ones(action_size)

    if state[0].observation.step == 0:
        return moves

    prev_action = str_to_action(state[player]['action'])
    invalid_action = Action.opposite(prev_action)
    moves[invalid_action.value - 1] = 0
    return moves


def select_action(pi, prev_action):
    '''
    Args:
        pi: np.array(4)
        prev_action: one of 0, 1, 2, 3
    Return:
        action: str
    '''
    if prev_action is not None:
        prev_action = str_to_action(prev_action)
        invalid_action = Action.opposite(prev_action)
        pi[invalid_action.value - 1] = 0
        pi /= np.sum(pi)
    action_num = np.random.choice(len(pi), p=pi) + 1
    action = Action(action_num).name
    return action


def agent(obs, config, prev_actions=None):
    if prev_actions is None:
        try:
            prev_actions = agent.prev_actions
        except AttributeError:
            agent.prev_actions = [None] * args.numAgents
            prev_actions = agent.prev_actions

    player = obs.index

    board = get_board(obs, prev_actions, args)
    board = get_player_board(board, player, args.numAgents)

    pi, _ = agent.net.predict(board)

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
