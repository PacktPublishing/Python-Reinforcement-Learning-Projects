MODEL_NUM_REGEX = "^\d{6}"
MODEL_NAME_REGEX = "^\d{6}(-\w+)+"

class HYPERPARAMS:
    BETA = 'beta'
    MOMENTUM = 'momentum'
    NUMSHAREDLAYERS = 'num_shared_layers'
    FC_WIDTH = 'fc_width'
    NUM_FILTERS = 'k'
    EPSILON = "epsilon"

class PATHS:
    MODELS_DIR = "models/"
    SELFPLAY_DIR = 'data/selfplay/'
    HOLDOUT_DIR = "data/holdout/"
    SGF_DIR = "data/sgf/"
    TRAINING_CHUNK_DIR = "data/training_chunks/"
    ESTIMATOR_WORKING_DIR = 'estimator_working_dir/'
    INITIAL_CHECKPOINT_NAME = "model.ckpt-1"

class FEATUREPARAMETERS:
    NUM_CHANNELS = 17
