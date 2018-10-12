import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

class ChildCNN(object):

    def __init__(self, cnn_dna, child_id, beta=1e-4, drop_rate=0.2, **kwargs):
        self.cnn_dna = self.process_raw_controller_output(cnn_dna)
        self.child_id = child_id
        self.beta = beta
        self.drop_rate = drop_rate
        self.is_training = tf.placeholder_with_default(True, shape=None, name="is_training_{}".format(self.child_id))
        self.num_classes = 10

    def process_raw_controller_output(self, output):
        """
        A helper function for preprocessing the output of the NASCell
        Args:
            output (numpy.ndarray) The output of the NASCell

        Returns:
            (list) The child network's architecture
        """
        output = output.ravel()
        cnn_dna = [list(output[x:x+4]) for x in range(0, len(output), 4)]
        return cnn_dna

    def build(self, input_tensor):
        """
        Method for creating the child neural network
        Args:
            input_tensor: The tensor which represents the input

        Returns:
            The tensor which represents the output logit (pre-softmax activation)

        """
        logger.info("DNA is: {}".format(self.cnn_dna))
        output = input_tensor
        for idx in range(len(self.cnn_dna)):
            # Get the configuration for the layer
            kernel_size, stride, num_filters, max_pool_size = self.cnn_dna[idx]
            with tf.name_scope("child_{}_conv_layer_{}".format(self.child_id, idx)):
                output = tf.layers.conv2d(output,
                        # Specify the number of filters the convolutional layer will output
                        filters=num_filters,
                        # This specifies the size (height, width) of the convolutional kernel
                        kernel_size=(kernel_size, kernel_size),
                        # The size of the stride of the kernel
                        strides=(stride, stride),
                        # We add padding to the image
                        padding="SAME",
                        # It is good practice to name your layers
                        name="conv_layer_{}".format(idx),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.beta))
                # We apply 2D max pooling on the output of the conv layer
                output = tf.layers.max_pooling2d(
                    output, pool_size=(max_pool_size, max_pool_size), strides=1,
                    padding="SAME", name="pool_out_{}".format(idx)
                )
                # Dropout to regularize the network further
                output = tf.layers.dropout(output, rate=self.drop_rate, training=self.is_training)

        # Lastly, we flatten the outputs and add a fully-connected layer
        with tf.name_scope("child_{}_fully_connected".format(self.child_id)):
            output = tf.layers.flatten(output, name="flatten")
            logits = tf.layers.dense(output, self.num_classes)

        return logits
