"""
Configuration store
"""
from constants import HYPERPARAMS

class GOPARAMETERS:
    N = 9
    WHITE = -1
    EMPTY = 0
    BLACK = 1
    FILL = 2
    KO = 3
    UNKNOWN = 4
    MISSING_GROUP_ID = -1
    COL_NAMES = 'ABCDEFGHJKLMNOPQRST'
    SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'

class GLOBAL_PARAMETER_STORE:
    # How many positions we should aggregate per 'chunk'.
    EXAMPLES_PER_RECORD = 10000
    # How many positions to draw from for our training window.
    # AGZ used the most recent 500k games, which, assuming 250 moves/game = 125M
    WINDOW_SIZE = 125000000
    # Number of positions to look at per generation
    EXAMPLES_PER_GENERATION = 2000000
    # Number of selfplay games
    NUM_SELFPLAY_GAMES = 100
    # Positions per batch
    TRAIN_BATCH_SIZE = 16
    # Number of games before the selfplay workers will stop
    MAX_GAMES_PER_GENERATION = 10000
    # Proportion of games to holdout from training per generation
    HOLDOUT = 0.05
    # Number of leaves to consider simultaneously
    SIMULTANEOUS_LEAVES = 8
    # Step boundaries for changing the learning rate
    BOUNDARIES = [int(1e6), int(2e6)]
    # Learning rates corresponding to boundaries
    LEARNING_RATE = [1e-2, 1e-3, 1e-4]
    SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[Minigo_sgfgenerator]RU[{ruleset}]
    SZ[{boardsize}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
    {game_moves})'''
    PROGRAM_IDENTIFIER = "AlphaGoZero"
    TEMPERATURE_CUTOFF = int((GOPARAMETERS.N * GOPARAMETERS.N) / 12)
    # TFRecords related parameters
    SHUFFLE_BUFFER_SIZE = int(2*1e4)
    CYCLE_LENGTH = 16
    BLOCK_LENGTH = 64
    # Number of MCTS readouts we do during selfplay
    SELFPLAY_READOUTS = 1600
    # Default resign threshold
    RESIGN_THRESHOLD = -0.90
    # Number of MCTS readouts we do during evaluation
    EVALUATION_READOUTS = 400
    # Number of games to play during evaluation
    EVALUATION_GAMES = 16
    # Buffer size for when validating model
    VALIDATION_BUFFER_SIZE = 1000
    # Number of global steps when validating model
    VALIDATION_NUMBER_OF_STEPS = 1000

class MCTSPARAMETERS:
    # 505 moves for 19x19, 113 for 9x9
    MAX_DEPTH = (GOPARAMETERS.N ** 2) * 1.4
    # Exploration constant
    c_PUCT = 1.38
    # Dirichlet noise, as a function of GOPARAMETERS.N
    DIRICHLET_NOISE = 0.03 * 361 / (GOPARAMETERS.N ** 2)

class AGENTPARAMETERS:
    SECONDS_PER_MOVE = 5

ALL_POSITIONS = [(i, j) for i in range(GOPARAMETERS.N) for j in range(GOPARAMETERS.N)]
NEIGHBORS = {(x, y): list(filter(lambda c: c[0] % GOPARAMETERS.N == c[0] and c[1] % GOPARAMETERS.N == c[1], [
    (x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_POSITIONS}
DIAGONALS = {(x, y): list(filter(lambda c: c[0] % GOPARAMETERS.N == c[0] and c[1] % GOPARAMETERS.N == c[1], [
    (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_POSITIONS}

"""
    k: number of filters (AlphaGoZero used 256). We use 128 by
    default for a 19x19 go board.
    fc_width: Dimensionality of the fully connected linear layer
    num_shared_layers: number of shared residual blocks.  AGZ used both 19
    and 39. Here we use 19 because it's faster to train.
    l2_strength: The L2 regularization parameter.
    momentum: The momentum parameter for training
"""
NETWORK_HYPERPARAMETERS = {
    HYPERPARAMS.NUM_FILTERS: 128,  # Width of each conv layer
    HYPERPARAMS.FC_WIDTH: 2 * 128,  # Width of each fully connected layer
    HYPERPARAMS.NUMSHAREDLAYERS: 19,  # Number of shared trunk layers
    HYPERPARAMS.BETA: 1e-4,  # Regularization strength
    HYPERPARAMS.MOMENTUM: 0.9,  # Momentum used in SGD
}
