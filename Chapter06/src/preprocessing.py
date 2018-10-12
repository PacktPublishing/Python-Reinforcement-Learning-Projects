import logging
import random

import numpy as np
import tensorflow as tf

from config import GOPARAMETERS, GLOBAL_PARAMETER_STORE
from constants import FEATUREPARAMETERS
from features import extract_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

def _one_hot(index):
    onehot = np.zeros([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32)
    onehot[index] = 1
    return onehot


def get_input_tensors(list_tf_records, buffer_size=GLOBAL_PARAMETER_STORE.SHUFFLE_BUFFER_SIZE):
    logger.info("Getting input data and tensors")
    dataset = process_tf_records(list_tf_records=list_tf_records,
                                 buffer_size=buffer_size)
    dataset = dataset.filter(lambda input_tensor: tf.equal(tf.shape(input_tensor)[0],
                                                           GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE))
    dataset = dataset.map(parse_batch_tf_example)
    logger.info("Finished parsing")
    return dataset.make_one_shot_iterator().get_next()


def create_dataset_from_selfplay(data_extracts):
    return (create_tf_train_example(extract_features(board_state), pi, result)
            for board_state, pi, result in data_extracts)


def shuffle_tf_examples(batch_size, records_to_shuffle):
    tf_dataset = process_tf_records(records_to_shuffle, batch_size=batch_size)
    iterator = tf_dataset.make_one_shot_iterator()
    next_dataset_batch = iterator.get_next()
    sess = tf.Session()
    while True:
        try:
            result = sess.run(next_dataset_batch)
            yield list(result)
        except tf.errors.OutOfRangeError:
            break


def create_tf_train_example(board_state, pi, result):
    board_state_as_tf_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[board_state.tostring()]))
    pi_as_tf_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[pi.tostring()]))
    value_as_tf_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[result]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': board_state_as_tf_feature,
        'pi': pi_as_tf_feature,
        'z': value_as_tf_feature
    }))

    return tf_example

def write_tf_examples(record_path, tf_examples, serialize=True):
    with tf.python_io.TFRecordWriter(record_path, options=TF_RECORD_CONFIG) as tf_record_writer:
        for tf_example in tf_examples:
            if serialize:
                tf_record_writer.write(tf_example.SerializeToString())
            else:
                tf_record_writer.write(tf_example)

def parse_batch_tf_example(example_batch):
    features = {
        'x': tf.FixedLenFeature([], tf.string),
        'pi': tf.FixedLenFeature([], tf.string),
        'z': tf.FixedLenFeature([], tf.float32),
    }
    parsed_tensors = tf.parse_example(example_batch, features)

    # Get the board state
    x = tf.cast(tf.decode_raw(parsed_tensors['x'], tf.uint8), tf.float32)
    x = tf.reshape(x, [GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE, GOPARAMETERS.N,
                       GOPARAMETERS.N, FEATUREPARAMETERS.NUM_CHANNELS])

    # Get the policy target, which is the distribution of possible moves
    # Each target is a vector of length of board * length of board + 1
    distribution_of_moves = tf.decode_raw(parsed_tensors['pi'], tf.float32)
    distribution_of_moves = tf.reshape(distribution_of_moves,
                                       [GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE, GOPARAMETERS.N * GOPARAMETERS.N + 1])

    # Get the result of the game
    # The result is simply a scalar
    result_of_game = parsed_tensors['z']
    result_of_game.set_shape([GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE])

    return (x, {'pi_label': distribution_of_moves, 'z_label': result_of_game})


def process_tf_records(list_tf_records, shuffle_records=True,
                       buffer_size=GLOBAL_PARAMETER_STORE.SHUFFLE_BUFFER_SIZE,
                       batch_size=GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE):

    if shuffle_records:
        random.shuffle(list_tf_records)

    list_dataset = tf.data.Dataset.from_tensor_slices(list_tf_records)

    tensors_dataset = list_dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x, compression_type='ZLIB'),
                                             cycle_length=GLOBAL_PARAMETER_STORE.CYCLE_LENGTH,
                                             block_length=GLOBAL_PARAMETER_STORE.BLOCK_LENGTH)
    tensors_dataset = tensors_dataset.repeat(1).shuffle(buffer_siz=buffer_size).batch(batch_size)

    return tensors_dataset

