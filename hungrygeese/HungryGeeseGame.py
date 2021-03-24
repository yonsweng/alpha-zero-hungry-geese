import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Observation, Action, adjacent_positions


class HungryGeeseGame():
    def __init__(self, env=None, rows=7, columns=11, hunger_rate=40, min_food=2, num_agents=4):
        if env is not None:
            self.env = env.clone()
        else:
            self.env = make(
                "hungry_geese",
                configuration={
                    "rows": rows,
                    "columns": columns,
                    "hunger_rate": hunger_rate,
                    "min_food": min_food
                },
                debug=False
            )
        self.config = Configuration(self.env.configuration)
        self.num_agents = num_agents
        self.action_size = len(Action)
        self.str_to_action = {'NORTH': Action.NORTH, 'EAST': Action.EAST, 'SOUTH': Action.SOUTH, 'WEST': Action.WEST}
        self.action_to_num = {'NORTH': 1, 'EAST': 2, 'SOUTH': 3, 'WEST': 4}
        self.prev_actions = None

    def get_init_obs(self):
        return Observation(self.env.reset(self.num_agents))

    def get_board(self, obs, player):
        return get_board

    def getBoardSize(self):
        return (self.config.rows, self.config.columns)

    def getActionSize(self):
        return self.action_size

    def getNextBoard(self, actions):
        self.prev_actions = actions
        obs, _, done, info = self.env.step(actions)
        obs = Observation(obs)
        reward = self.getReward(obs) if done else None
        board = self.getBoard(obs, 0)
        return board, reward, done, info

    def getValidMoves(self, player):
        prev_action = self.str_to_action[self.prev_actions[player]]
        oppo_action = Action.opposite(prev_action).value
        valids = [1] * len(Action)
        valids[oppo_action - 1] = 0
        return np.array(valids)

    def getReward(self, obs, player) -> float:
        rank = 1
        for i, goose in enumerate(obs.geese):
            if i != player:
                if len(goose) > len(obs.geese[player]):
                    rank += 1
                elif len(goose) == len(obs.geese[player]):
                    rank += 0.5
        return (self.num_agents + 1 - 2 * rank) / (self.num_agents - 1)

    def get_board(self, obs, player):
        '''
        Channels of state:
            geese curr_heads  0,  1,  2,  3
            geese bodies      4,  5,  6,  7
            geese tips        8,  9, 10, 11
            geese prev_heads 12, 13, 14, 15
            food 16
        '''
        board = np.zeros((self.num_agents * 4 + 1, self.config.rows * self.config.columns), np.uint8)

        for i, goose in enumerate(obs.geese):
            # head position
            for head_pos in goose[:1]:
                board[0 + (i - player) % self.num_agents, head_pos] = 1

            # tip position
            for tip_pos in goose[-1:]:
                board[4 + (i - player) % self.num_agents, tip_pos] = 1

            # whole position
            for body_pos in goose[1:]:
                board[8 + (i - player) % self.num_agents, body_pos] = 1

            # previous head position
            for head_pos in goose[:1]:
                if obs.step > 0:
                    opposite_action = Action.opposite(self.str_to_action[prev_actions[i]])
                    prev_head_pos = adjacent_positions(head_pos, config.columns, config.rows)[opposite_action.value - 1]

                    board[12 + (i - player) % self.num_agents, prev_head_pos] = 1

        for food_pos in obs.food:
            board[-1, food_pos] = 1

        return board.reshape(-1, 7, 11)

    def getCanonicalState(self, state, player):
        new_state = state
        indices = np.arange(0, self.num_agents * 4, self.num_agents)
        tmp = new_state[indices]
        new_state[indices] = new_state[indices + player]
        new_state[indices + player] = tmp
        return new_state

    def getSymmetries(self, board, player, pi):
        state = self.boardToState(board)
        sym = []

        # original
        sym.append((state, pi))

        # 180 degree turn
        tmp_state = np.rot90(state, 2, (1, 2))
        tmp_pi = [*pi[2:], *pi[:2]]
        sym.append((tmp_state, tmp_pi))

        # flip ud
        tmp_state = np.flip(state, 1)
        tmp_pi = pi.copy()
        tmp = tmp_pi[0]
        tmp_pi[0] = tmp_pi[2]
        tmp_pi[2] = tmp
        sym.append((tmp_state, tmp_pi))

        # flip lr
        tmp_state = np.flip(state, 2)
        tmp_pi = pi.copy()
        tmp = tmp_pi[1]
        tmp_pi[1] = tmp_pi[3]
        tmp_pi[3] = tmp
        sym.append((tmp_state, tmp_pi))

        return sym
