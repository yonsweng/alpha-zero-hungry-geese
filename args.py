from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 30,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 500000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 60,         # Number of games to play during arena play to determine if new net will be accepted.
    'numProcesses': 15,
    'maxDepth': 4,
    'cpuct': 1,

    'cuda': torch.cuda.is_available(),
    'n_gpus': torch.cuda.device_count(),
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,

    'boardSize': (7, 11),
    'actionSize': len(Action),
    'numAgents': 4
})
