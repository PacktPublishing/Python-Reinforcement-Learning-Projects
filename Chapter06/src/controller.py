import argparse
import logging
import os
import random
import socket
import sys
import time

import argh
import tensorflow as tf
from tensorflow import gfile
from tqdm import tqdm

import alphagozero_agent
import network
import preprocessing
from config import GLOBAL_PARAMETER_STORE
from constants import PATHS
from alphagozero_agent import play_match
from network import PolicyValueNetwork
from utils import logged_timer as timer
from utils import print_flags, generate, detect_model_name, detect_model_version

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
)

logger = logging.getLogger(__name__)

def get_models():
    """
    Get all model versions
    """
    all_models = gfile.Glob(os.path.join(PATHS.MODELS_DIR, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_versionbers_names = sorted([
        (detect_model_version(m), detect_model_name(m))
        for m in model_filenames])
    return model_versionbers_names

def get_latest_model():
    """
    Get the latest model

    Returns:
        Tuple of <int, str>, or <model_version, model_name>
    """
    return get_models()[-1]


def get_model(model_version):
    models = {k: v for k, v in get_models()}
    if not model_version in models:
        raise ValueError("Model {} not found!".format(model_version))
    return models[model_version]

def initialize_random_model():
    bootstrap_name = generate(0)
    bootstrap_model_path = os.path.join(PATHS.MODELS_DIR, bootstrap_name)
    logger.info("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        PATHS.ESTIMATOR_WORKING_DIR, bootstrap_model_path))
    os.makedirs(PATHS.ESTIMATOR_WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(bootstrap_model_path), exist_ok=True)
    network.initialize_random_model(PATHS.ESTIMATOR_WORKING_DIR)
    network.export_latest_checkpoint_model(PATHS.ESTIMATOR_WORKING_DIR, bootstrap_model_path)


def selfplay():
    _, model_name = get_latest_model()
    try:
        games = gfile.Glob(os.path.join(PATHS.SELFPLAY_DIR, model_name, '*.zz'))
        if len(games) > GLOBAL_PARAMETER_STORE.MAX_GAMES_PER_GENERATION:
            logger.info("{} has enough games ({})".format(model_name, len(games)))
            time.sleep(600)
            sys.exit(1)
    except:
        pass

    for game_idx in range(GLOBAL_PARAMETER_STORE.NUM_SELFPLAY_GAMES):
        logger.info('================================================')
        logger.info("Playing game {} with model {}".format(game_idx, model_name))
        logger.info('================================================')
        model_save_path = os.path.join(PATHS.MODELS_DIR, model_name)
        game_output_dir = os.path.join(PATHS.SELFPLAY_DIR, model_name)
        game_holdout_dir = os.path.join(PATHS.HOLDOUT_DIR, model_name)
        sgf_dir = os.path.join(PATHS.SGF_DIR, model_name)

        clean_sgf = os.path.join(sgf_dir, 'clean')
        full_sgf = os.path.join(sgf_dir, 'full')
        os.makedirs(clean_sgf, exist_ok=True)
        os.makedirs(full_sgf, exist_ok=True)
        os.makedirs(game_output_dir, exist_ok=True)
        os.makedirs(game_holdout_dir, exist_ok=True)

        with timer("Loading weights from %s ... " % model_save_path):
            network = PolicyValueNetwork(model_save_path)

        with timer("Playing game"):
            agent = alphagozero_agent.play_against_self(network, GLOBAL_PARAMETER_STORE.SELFPLAY_READOUTS)

        output_name = '{}-{}'.format(int(time.time()), socket.gethostname())
        game_play = agent.extract_data()
        with gfile.GFile(os.path.join(clean_sgf, '{}.sgf'.format(output_name)), 'w') as f:
            f.write(agent.to_sgf(use_comments=False))
        with gfile.GFile(os.path.join(full_sgf, '{}.sgf'.format(output_name)), 'w') as f:
            f.write(agent.to_sgf())

        tf_examples = preprocessing.create_dataset_from_selfplay(game_play)

        # We reserve 5% of games played for validation
        holdout = random.random() < GLOBAL_PARAMETER_STORE.HOLDOUT
        if holdout:
            to_save_dir = game_holdout_dir
        else:
            to_save_dir = game_output_dir
        tf_record_path = os.path.join(to_save_dir, "{}.tfrecord.zz".format(output_name))

        preprocessing.write_tf_examples(tf_record_path, tf_examples)


def aggregate():
    logger.info("Gathering game results")

    os.makedirs(PATHS.TRAINING_CHUNK_DIR, exist_ok=True)
    os.makedirs(PATHS.SELFPLAY_DIR, exist_ok=True)
    models = [model_dir.strip('/')
              for model_dir in sorted(gfile.ListDirectory(PATHS.SELFPLAY_DIR))[-50:]]

    with timer("Finding existing tfrecords..."):
        model_gamedata = {
            model: gfile.Glob(
                os.path.join(PATHS.SELFPLAY_DIR, model, '*.zz'))
            for model in models
        }
    logger.info("Found %d models" % len(models))
    for model_name, record_files in sorted(model_gamedata.items()):
        logger.info("    %s: %s files" % (model_name, len(record_files)))

    meta_file = os.path.join(PATHS.TRAINING_CHUNK_DIR, 'meta.txt')
    try:
        with gfile.GFile(meta_file, 'r') as f:
            already_processed = set(f.read().split())
    except tf.errors.NotFoundError:
        already_processed = set()

    num_already_processed = len(already_processed)

    for model_name, record_files in sorted(model_gamedata.items()):
        if set(record_files) <= already_processed:
            continue
        logger.info("Gathering files for %s:" % model_name)
        for i, example_batch in enumerate(
                tqdm(preprocessing.shuffle_tf_examples(GLOBAL_PARAMETER_STORE.EXAMPLES_PER_RECORD, record_files))):
            output_record = os.path.join(PATHS.TRAINING_CHUNK_DIR,
                                         '{}-{}.tfrecord.zz'.format(model_name, str(i)))
            preprocessing.write_tf_examples(
                output_record, example_batch, serialize=False)
        already_processed.update(record_files)

    logger.info("Processed %s new files" %
          (len(already_processed) - num_already_processed))
    with gfile.GFile(meta_file, 'w') as f:
        f.write('\n'.join(sorted(already_processed)))

def train():
    model_version, model_name = get_latest_model()
    logger.info("Training on gathered game data, initializing from {}".format(model_name))
    new_model_name = generate(model_version + 1)
    logger.info("New model will be {}".format(new_model_name))
    save_file = os.path.join(PATHS.MODELS_DIR, new_model_name)

    try:
        logger.info("Getting tf_records")
        tf_records = sorted(gfile.Glob(os.path.join(PATHS.TRAINING_CHUNK_DIR, '*.tfrecord.zz')))
        tf_records = tf_records[
                     -1 * (GLOBAL_PARAMETER_STORE.WINDOW_SIZE // GLOBAL_PARAMETER_STORE.EXAMPLES_PER_RECORD):]

        print("Training from:", tf_records[0], "to", tf_records[-1])

        with timer("Training"):
            network.train(PATHS.ESTIMATOR_WORKING_DIR, tf_records, model_version+1)
            network.export_latest_checkpoint_model(PATHS.ESTIMATOR_WORKING_DIR, save_file)

    except:
        logger.info("Got an error training")
        logging.exception("Train error")


def validate(model_version=None, validate_name=None):
    if model_version is None:
        model_version, model_name = get_latest_model()
    else:
        model_version = int(model_version)
        model_name = get_model(model_version)

    models = list(
        filter(lambda num_name: num_name[0] < (model_version - 1), get_models()))

    if len(models) == 0:
        logger.info('Not enough models, including model N for validation')
        models = list(
            filter(lambda num_name: num_name[0] <= model_version, get_models()))
    else:
        logger.info('Validating using data from following models: {}'.format(models))

    tf_record_dirs = [os.path.join(PATHS.HOLDOUT_DIR, pair[1])
                    for pair in models[-5:]]

    working_dir = PATHS.ESTIMATOR_WORKING_DIR
    checkpoint_name = os.path.join(PATHS.MODELS_DIR, model_name)

    tf_records = []
    with timer("Building lists of holdout files"):
        for record_dir in tf_record_dirs:
            tf_records.extend(gfile.Glob(os.path.join(record_dir, '*.zz')))

    with timer("Validating from {} to {}".format(os.path.basename(tf_records[0]), os.path.basename(tf_records[-1]))):
        network.validate(working_dir, tf_records, checkpoint_path=checkpoint_name, name=validate_name)

def evaluate(black_model, white_model):
    os.makedirs(PATHS.SGF_DIR, exist_ok=True)

    with timer("Loading weights"):
        black_net = network.PolicyValueNetwork(black_model)
        white_net = network.PolicyValueNetwork(white_model)

    with timer("Playing {} games".format(GLOBAL_PARAMETER_STORE.EVALUATION_GAMES)):
        play_match(black_net, white_net, GLOBAL_PARAMETER_STORE.EVALUATION_GAMES,
                   GLOBAL_PARAMETER_STORE.EVALUATION_READOUTS, PATHS.SGF_DIR)


parser = argparse.ArgumentParser()

argh.add_commands(parser, [train, selfplay, aggregate, evaluate, initialize_random_model, validate])

if __name__ == '__main__':
    print_flags()
    argh.dispatch(parser)
