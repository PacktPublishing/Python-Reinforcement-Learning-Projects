'''
Created on 29 May 2017

@author: ywz
'''
import numpy
import tensorflow as tf


def leaky_relu(x, leak=0.0, name="lrelu"):
    return tf.maximum(leak * x, x, name=name)

def add_regularization(var, weight):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), weight, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

def get_variable_on_cpu(shape, initializer, name, dtype=tf.float32, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(shape=shape, initializer=initializer, 
                              dtype=dtype, name=name, trainable=trainable)
    return var

def HeUniform(shape):
    
    if len(shape) > 2:
        w = shape[0]
        h = shape[1]
        input_channels  = shape[2]
        d = 1.0 / numpy.sqrt(input_channels * w * h)
    else:
        d = 1.0 / numpy.sqrt(shape[0])
    
    init_W = tf.random_uniform_initializer(-d, d)
    init_b = tf.random_uniform_initializer(-d, d)
    return init_W, init_b

def conv2d(x, output_dim, kernel=(5, 5), stride=(2, 2), 
           activation=tf.nn.relu, init_W=None, init_b=None, name='conv', padding='VALID'):
    
    assert len(x.get_shape().as_list()) == 4
    shape = (kernel[0], kernel[1], x.get_shape().as_list()[-1], output_dim)
    _W, _b = HeUniform(shape)
    if init_W is None: init_W = _W
    if init_b is None: init_b = _b

    with tf.variable_scope(name):
        W = get_variable_on_cpu(shape=shape, initializer=init_W, dtype=tf.float32, name='weight')
        b = get_variable_on_cpu(shape=(output_dim,), initializer=init_b, dtype=tf.float32, name='bias')
        
        conv = tf.nn.conv2d(input=x, filter=W, strides=(1, stride[0], stride[1], 1), padding=padding)
        if activation:
            conv = activation(tf.nn.bias_add(conv, b))
        else:
            conv = tf.nn.bias_add(conv, b)
    
    return conv

def linear(x, output_dim, activation=tf.nn.relu, init_W=None, init_b=None, name='linear'):
    
    if len(x.get_shape().as_list()) > 2:
        shape = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, numpy.prod(shape[1:])))

    shape = (x.get_shape().as_list()[-1], output_dim)
    _W, _b = HeUniform(shape)
    if init_W is None: init_W = _W
    if init_b is None: init_b = _b

    with tf.variable_scope(name):
        W = get_variable_on_cpu(shape=shape, initializer=init_W, dtype=tf.float32, name='weight')
        b = get_variable_on_cpu(shape=(output_dim,), initializer=init_b, dtype=tf.float32, name='bias')
        
        linear = tf.matmul(x, W) + b
        if activation:
            linear = activation(linear)
    
    return linear


