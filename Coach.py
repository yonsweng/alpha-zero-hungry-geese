import logging
import os
import sys
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from tqdm import tqdm

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

from Arena import Arena, playGames
from MCTS import MCTS
from agent import *

log = logging.getLogger(__name__)


def executeEpisode(env, nnet, args, rank):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
    """
    trainExamples = []
    episodeStep = 0
    prev_actions = [None] * args.numAgents

    obs = env.reset(args.numAgents)[0].observation

    while True:
        episodeStep += 1

        # temp = int(episodeStep < args.tempThreshold)
        temp = 1

        mcts = MCTS(nnet, args, rank)
        pi = mcts.getActionProb(env, prev_actions[:], rank, temp)
        board = get_board(obs, prev_actions, args)

        sym = get_symmetries(board, 0, args.numAgents, pi)
        for b, p in sym:
            trainExamples.append([b, p, None])

        actions = []

        # player 0's action
        action = select_action(pi, prev_actions[0])
        actions.append(action)

        # other players' action
        pis, _ = nnet.predicts(board, rank % args.n_gpus)
        for i, pi in enumerate(pis[1:], 1):
            action = select_action(pi, prev_actions[i])
            actions.append(action)

        obs = env.step(actions)[0].observation

        r = get_reward(obs, 0, args.numAgents)
        if r is not None:
            break

        prev_actions = actions

    return [(x[0], x[1], r) for x in trainExamples]


def rollout(rank, nnet, args, examples):
    env = make('hungry_geese', configuration={'rows': args.boardSize[0], 'columns': args.boardSize[1]})
    for _ in tqdm(range(args.numEps), desc="Self Play"):
        examples += executeEpisode(env, nnet, args, rank)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.args)  # the competitor network
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')

            # copy the model for multi-GPU training
            self.nnet.copy_net()

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # for parallel rollouts
                with mp.Manager() as manager:
                    examples = manager.list()
                    mp.spawn(rollout, args=(self.nnet, self.args, examples), nprocs=self.args.numProcesses)
                    iterationTrainExamples.extend(list(examples))

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            if os.path.exists('best.pth.tar'):
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            self.nnet.train(trainExamples)

            # copy the model for multi-GPU training
            self.pnet.copy_net()

            log.info('PITTING AGAINST PREVIOUS VERSION')
            avg_rwd, avg_len = playGames(self.pnet, self.nnet, self.args.arenaCompare, self.args)

            log.info('AVG REWARD : %.2f, AVG LENGTH : %.2f' % (avg_rwd, avg_len))
            if avg_rwd < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
