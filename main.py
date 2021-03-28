import logging
import coloredlogs
import torch.multiprocessing as mp
from Coach import Coach
from hungrygeese.pytorch.NNet import NNetWrapper as nn
from utils import *
from args import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main():
    log.info('Loading %s...', nn.__name__)
    nnet = nn(args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
