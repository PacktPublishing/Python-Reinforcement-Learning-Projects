"""
Textbook code that implements a CNN using TensorFlow
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils

class SimpleCNN(object):

    def __init__(self, learning_rate, num_epochs, beta, batch_size):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.beta = beta
        self.batch_size = batch_size
        self.save_dir = "saves"
        self.logs_dir = "logs"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "simple_cnn")
        self.logs_path = os.path.join(self.logs_dir, "simple_cnn")

    def build(self, input_tensor, num_classes):
        """
        Builds a convolutional neural network according to the input shape and the number of classes.
        Architecture is fixed.

        Args:
            input_tensor: Tensor of the input
            num_classes: (int) number of classes

        Returns:
            The output logits before softmax
        """
        with tf.name_scope("input_placeholders"):
            self.is_training = tf.placeholder_with_default(True, shape=(), name="is_training")

        with tf.name_scope("convolutional_layers"):
            conv_1 = tf.layers.conv2d(
                input_tensor,
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="conv_1")
            conv_2 = tf.layers.conv2d(
                conv_1,
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="conv_2")
            pool_3 = tf.layers.max_pooling2d(
                conv_2,
                pool_size=(2, 2),
                strides=1,
                padding="SAME",
                name="pool_3"
            )
            drop_4 = tf.layers.dropout(pool_3, training=self.is_training, name="drop_4")

            conv_5 = tf.layers.conv2d(
                drop_4,
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="conv_5")
            conv_6 = tf.layers.conv2d(
                conv_5,
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="conv_6")
            pool_7 = tf.layers.max_pooling2d(
                conv_6,
                pool_size=(2, 2),
                strides=1,
                padding="SAME",
                name="pool_7"
            )
            drop_8 = tf.layers.dropout(pool_7, training=self.is_training, name="drop_8")

        with tf.name_scope("fully_connected_layers"):
            flattened = tf.layers.flatten(drop_8, name="flatten")
            fc_9 = tf.layers.dense(
                flattened,
                units=1024,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="fc_9"
            )
            drop_10 = tf.layers.dropout(fc_9, training=self.is_training, name="drop_10")
            logits = tf.layers.dense(
                drop_10,
                units=num_classes,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta),
                name="logits"
            )

        return logits

    def _create_tf_dataset(self, x, y):
        dataset = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(x),
                tf.data.Dataset.from_tensor_slices(y)
            )).shuffle(50).repeat().batch(self.batch_size)
        return dataset

    def _log_loss_and_acc(self, epoch, loss, acc, suffix):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss_{}".format(suffix), simple_value=float(loss)),
            tf.Summary.Value(tag="acc_{}".format(suffix), simple_value=float(acc))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Trains a CNN on given data

        Args:
            numpy.ndarrays representing data and labels respectively
        """
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()

            train_dataset = self._create_tf_dataset(X_train, y_train)
            valid_dataset = self._create_tf_dataset(X_valid, y_valid)

            # Create a generic iterator
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
            next_tensor_batch = iterator.get_next()

            # Separate training and validation set init ops
            train_init_ops = iterator.make_initializer(train_dataset)
            valid_init_ops = iterator.make_initializer(valid_dataset)

            input_tensor, labels = next_tensor_batch

            num_classes = y_train.shape[1]

            # Build the network
            logits = self.build(input_tensor=input_tensor, num_classes=num_classes)
            logger.info('Built network')

            prediction = tf.nn.softmax(logits, name="predictions")
            loss_ops = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits), name="loss")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_ops = optimizer.minimize(loss_ops)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1), name="correct")
            accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

            initializer = tf.global_variables_initializer()

            # Training
            logger.info('Initializing all variables')
            sess.run(initializer)
            logger.info('Initialized all variables')

            sess.run(train_init_ops)
            logger.info('Initialized dataset iterator')
            self.saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=graph)

            logger.info("Training CNN for {} epochs".format(self.num_epochs))
            for epoch_idx in range(1, self.num_epochs+1):
                loss, _, accuracy = sess.run([
                    loss_ops, train_ops, accuracy_ops
                ])
                self._log_loss_and_acc(epoch_idx, loss, accuracy, "train")

                if epoch_idx % 10 == 0:
                    sess.run(valid_init_ops)
                    valid_loss, valid_accuracy = sess.run([
                        loss_ops, accuracy_ops
                    ], feed_dict={self.is_training: False})
                    logger.info("=====================> Epoch {}".format(epoch_idx))
                    logger.info("\tTraining accuracy: {:.3f}".format(accuracy))
                    logger.info("\tTraining loss: {:.6f}".format(loss))
                    logger.info("\tValidation accuracy: {:.3f}".format(valid_accuracy))
                    logger.info("\tValidation loss: {:.6f}".format(valid_loss))
                    self._log_loss_and_acc(epoch_idx, valid_loss, valid_accuracy, "valid")

                # Create a checkpoint every epoch
                self.saver.save(sess, self.save_path)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Loading Fashion MNIST data")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    logger.info('Shape of training data:')
    logger.info('Train: {}'.format(X_train.shape))
    logger.info('Test: {}'.format(X_test.shape))

    logger.info('Adding channel axis to the data')
    X_train = X_train[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]

    logger.info("Simple transformation by dividing pixels by 255")
    X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    num_classes = len(np.unique(y_train))

    logger.info("Turning ys into one-hot encodings")
    y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

    cnn_params = {
        "learning_rate": 3e-4,
        "num_epochs": 100,
        "beta": 1e-3,
        "batch_size": 32
    }

    logger.info('Initializing CNN')
    simple_cnn = SimpleCNN(**cnn_params)
    logger.info('Training CNN')
    simple_cnn.fit(X_train=X_train,
                   X_valid=X_test,
                   y_train=y_train,
                   y_valid=y_test)
