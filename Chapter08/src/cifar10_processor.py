import logging

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils

logger = logging.getLogger(__name__)

def _create_tf_dataset(x, y, batch_size):
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x),
                                tf.data.Dataset.from_tensor_slices(y))).shuffle(500).repeat().batch(batch_size)

def get_tf_datasets_from_numpy(batch_size, validation_split=0.1):
    """
    Main function getting tf.Data.datasets for training, validation, and testing

    Args:
        batch_size (int): Batch size
        validation_split (float): Split for partitioning training and validation sets. Between 0.0 and 1.0.
    """
    # Load data from keras datasets api
    (X, y), (X_test, y_test) = cifar10.load_data()

    logger.info("Dividing pixels by 255")
    X = X / 255.
    X_test = X_test / 255.

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y = y.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Turn labels into onehot encodings
    if y.shape[1] != 10:
        y = np_utils.to_categorical(y, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    logger.info("Loaded data from keras")

    split_idx = int((1.0 - validation_split) * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_valid, y_valid = X[split_idx:], y[split_idx:]

    train_dataset = _create_tf_dataset(X_train, y_train, batch_size)
    valid_dataset = _create_tf_dataset(X_valid, y_valid, batch_size)
    test_dataset = _create_tf_dataset(X_test, y_test, batch_size)

    # Get the batch sizes for the train, valid, and test datasets
    num_train_batches = int(X_train.shape[0] // batch_size)
    num_valid_batches = int(X_valid.shape[0] // batch_size)
    num_test_batches = int(X_test.shape[0] // batch_size)

    return train_dataset, valid_dataset, test_dataset, num_train_batches, num_valid_batches, num_test_batches
