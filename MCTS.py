import logging
import math

import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from agent import *

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """
    def __init__(self, nnet, args, rank):
        self.nnet = nnet
        self.args = args
        self.cuda_num = rank % args.n_gpus

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, env, prev_actions, rank, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for i in range(self.args.numMCTSSims):
            self.search(env.clone(), prev_actions[:], self.args.maxDepth, rank)

        obs = env.state[0].observation
        board = get_board(obs, prev_actions, self.args)
        s = bytes(obs.step) + board.tostring()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.args.actionSize)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, env, prev_actions, remaining, rank):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        state = env.state
        obs = state[0].observation
        board = get_board(obs, prev_actions, self.args)
        s = bytes(obs.step) + board.tostring()

        if s not in self.Es:
            self.Es[s] = get_reward(obs, 0, self.args.numAgents)
        if self.Es[s] is not None:
            # terminal node
            return self.Es[s]

        # predict players' action
        actions = []
        pis, vs = self.nnet.predicts(board, rank % self.args.n_gpus)
        for i, pi in enumerate(pis):
            action = select_action(pi, prev_actions[i])
            actions.append(action)
        v = vs[0]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = pis[0], v
            valids = get_valid_moves(state, 0, self.args.actionSize)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # return v

        if remaining == 0:
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.args.actionSize):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        actions[0] = Action(a + 1).name

        env.step(actions)

        v = self.search(env, actions, remaining - 1, rank)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
