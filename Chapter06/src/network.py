import functools
import logging
import os.path

import tensorflow as tf

import features
import preprocessing
import utils
from config import GLOBAL_PARAMETER_STORE, GOPARAMETERS
from constants import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_partial_bn_layer(params):
    return functools.partial(tf.layers.batch_normalization,
        momentum=params["momentum"],
        epsilon=params["epsilon"],
        fused=params["fused"],
        center=params["center"],
        scale=params["scale"],
        training=params["training"]
    )

def create_partial_res_layer(inputs, partial_bn_layer, partial_conv2d_layer):
    output_1 = partial_bn_layer(partial_conv2d_layer(inputs))
    output_2 = tf.nn.relu(output_1)
    output_3 = partial_bn_layer(partial_conv2d_layer(output_2))
    output_4 = tf.nn.relu(tf.add(inputs, output_3))
    return output_4

def softmax_cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels['pi_label']))

def mean_squared_loss(output_value, labels):
    return tf.reduce_mean(tf.square(output_value - labels['z_label']))

def get_losses(logits, output_value, labels):
    ce_loss = softmax_cross_entropy_loss(logits, labels)
    mse_loss = mean_squared_loss(output_value, labels)
    return ce_loss, mse_loss

def create_metric_ops(labels, output_policy, loss_policy, loss_value, loss_l2, loss_total):
    return {'accuracy': tf.metrics.accuracy(labels=labels['pi_label'], predictions=output_policy, name='accuracy'),
            'loss_policy': tf.metrics.mean(loss_policy),
            'loss_value': tf.metrics.mean(loss_value),
            'loss_l2': tf.metrics.mean(loss_l2),
            'loss_total': tf.metrics.mean(loss_total)}

class PolicyValueNetwork():

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.params = utils.parse_parameters(**kwargs)
        self.build_network()

    def build_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(graph=tf.Graph(), config=config)

    def build_network(self):
        self.sess = self.build_session()

        with self.sess.graph.as_default():
            features, labels = get_inference_input()
            model_spec = generate_network_specifications(features, labels,
                                                         tf.estimator.ModeKeys.PREDICT, self.params)
            self.inference_input = features
            self.inference_output = model_spec.predictions
            if self.model_path is not None:
                self.load_network_weights(self.model_path)
            else:
                self.sess.run(tf.global_variables_initializer())

    def load_network_weights(self, save_file):
        tf.train.Saver().restore(self.sess, save_file)

    def predict_on_single_board_state(self, position):
        probs, values = self.predict_on_multiple_board_states([position])
        prob = probs[0]
        value = values[0]
        return prob, value

    def predict_on_multiple_board_states(self, positions):
        symmetries, processed = utils.shuffle_feature_symmetries(list(map(features.extract_features, positions)))
        network_outputs = self.sess.run(self.inference_output, feed_dict={self.inference_input: processed})
        action_probs, value_pred = network_outputs['policy_output'], network_outputs['value_output']
        action_probs = utils.invert_policy_symmetries(symmetries, action_probs)
        return action_probs, value_pred

def get_inference_input():
    return (tf.placeholder(tf.float32,
                           [None, GOPARAMETERS.N, GOPARAMETERS.N, FEATUREPARAMETERS.NUM_CHANNELS],
                           name='board_state_features'),
            {'pi_label': tf.placeholder(tf.float32, [None, GOPARAMETERS.N * GOPARAMETERS.N + 1]),
             'z_label': tf.placeholder(tf.float32, [None])})


def generate_network_specifications(features, labels, mode, params, config=None):

    batch_norm_params = {"epsilon": 1e-5, "fused": True, "center": True, "scale": True, "momentum": 0.997,
                         "training": mode==tf.estimator.ModeKeys.TRAIN}

    # The main network that is shared by both policy and value networks
    with tf.name_scope("shared_layers"):
        partial_bn_layer = create_partial_bn_layer(batch_norm_params)
        partial_conv2d_layer = functools.partial(tf.layers.conv2d,
            filters=params[HYPERPARAMS.NUM_FILTERS], kernel_size=[3, 3], padding="same")
        partial_res_layer = functools.partial(create_partial_res_layer, batch_norm=partial_bn_layer,
                                              conv2d=partial_conv2d_layer)

        output_shared = tf.nn.relu(partial_bn_layer(partial_conv2d_layer(features)))

        for i in range(params[HYPERPARAMS.NUMSHAREDLAYERS]):
            output_shared = partial_res_layer(output_shared)

    # Implement the policy network
    with tf.name_scope("policy_network"):
        conv_p_output = tf.nn.relu(partial_bn_layer(partial_conv2d_layer(output_shared, filters=2,
                                                                              kernel_size=[1, 1]),
                                                                              center=False, scale=False))
        logits = tf.layers.dense(tf.reshape(conv_p_output, [-1, GOPARAMETERS.N * GOPARAMETERS.N * 2]),
                                 units=GOPARAMETERS.N * GOPARAMETERS.N + 1)
        output_policy = tf.nn.softmax(logits,
                                      name='policy_output')

    # Implement the value network
    with tf.name_scope("value_network"):
        conv_v_output = tf.nn.relu(partial_bn_layer(partial_conv2d_layer(output_shared, filters=1, kernel_size=[1, 1]),
            center=False, scale=False))
        fc_v_output = tf.nn.relu(tf.layers.dense(
            tf.reshape(conv_v_output, [-1, GOPARAMETERS.N * GOPARAMETERS.N]),
            params[HYPERPARAMS.FC_WIDTH]))
        fc_v_output = tf.layers.dense(fc_v_output, 1)
        fc_v_output = tf.reshape(fc_v_output, [-1])
        output_value = tf.nn.tanh(fc_v_output, name='value_output')

    # Implement the loss functions
    with tf.name_scope("loss_functions"):
        loss_policy, loss_value = get_losses(logits=logits,
                                             output_value=output_value,
                                             labels=labels)
        loss_l2 = params[HYPERPARAMS.BETA] * tf.add_n([tf.nn.l2_loss(v)
            for v in tf.trainable_variables() if not 'bias' in v.name])
        loss_total = loss_policy + loss_value + loss_l2

    # Steps and operations for training
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.piecewise_constant(global_step, GLOBAL_PARAMETER_STORE.BOUNDARIES,
                                                GLOBAL_PARAMETER_STORE.LEARNING_RATE)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = tf.train.MomentumOptimizer(learning_rate,
                    params[HYPERPARAMS.MOMENTUM]).minimize(loss_total, global_step=global_step)

    metric_ops = create_metric_ops(labels=labels,
                                   output_policy=output_policy,
                                   loss_policy=loss_policy,
                                   loss_value=loss_value,
                                   loss_l2=loss_l2,
                                   loss_total=loss_total)

    for metric_name, metric_op in metric_ops.items():
        tf.summary.scalar(metric_name, metric_op[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'policy_output': output_policy,
            'value_output': output_value,
        },
        loss=loss_total,
        train_op=train_op,
        eval_metric_ops=metric_ops,
    )

def get_estimator(estimator_dir, **kwargs):
    params = utils.parse_parameters(**kwargs)
    return tf.estimator.Estimator(generate_network_specifications, model_dir=estimator_dir, params=params)

def initialize_random_model(estimator_dir, **kwargs):
    sess = tf.Session(graph=tf.Graph())
    params = utils.parse_parameters(**kwargs)
    initial_model_path = os.path.join(estimator_dir, PATHS.INITIAL_CHECKPOINT_NAME)

    # Create the first model, where all we do is initialize random weights and immediately write them to disk
    with sess.graph.as_default():
        features, labels = get_inference_input()
        generate_network_specifications(features, labels, tf.estimator.ModeKeys.PREDICT, params)
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().save(sess, initial_model_path)

def export_latest_checkpoint_model(estimator_dir, model_path):
    estimator = tf.estimator.Estimator(generate_network_specifications, model_dir=estimator_dir, params='ignored')
    latest_checkpoint = estimator.latest_checkpoint()
    all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
    for filename in all_checkpoint_files:
        suffix = filename.partition(latest_checkpoint)[2]
        destination_path = model_path + suffix
        logger.info("Copying {} to {}".format(filename, destination_path))
        tf.gfile.Copy(filename, destination_path)


def train(estimator_dir, tf_records, model_version, **kwargs):
    """
    Main training function for the PolicyValueNetwork
    Args:
        estimator_dir (str): Path to the estimator directory
        tf_records (list): A list of TFRecords from which we parse the training examples
        model_version (int): The version of the model
    """
    model = get_estimator(estimator_dir, **kwargs)
    logger.info("Training model version: {}".format(model_version))
    max_steps = model_version * GLOBAL_PARAMETER_STORE.EXAMPLES_PER_GENERATION // \
                GLOBAL_PARAMETER_STORE.TRAIN_BATCH_SIZE
    model.train(input_fn=lambda: preprocessing.get_input_tensors(list_tf_records=tf_records),
                max_steps=max_steps)
    logger.info("Trained model version: {}".format(model_version))


def validate(estimator_dir, tf_records, checkpoint_path=None, **kwargs):
    model = get_estimator(estimator_dir, **kwargs)
    if checkpoint_path is None:
        checkpoint_path = model.latest_checkpoint()
    model.evaluate(input_fn=lambda: preprocessing.get_input_tensors(
        list_tf_records=tf_records,
        buffer_size=GLOBAL_PARAMETER_STORE.VALIDATION_BUFFER_SIZE),
                   steps=GLOBAL_PARAMETER_STORE.VALIDATION_NUMBER_OF_STEPS,
                   checkpoint_path=checkpoint_path)
