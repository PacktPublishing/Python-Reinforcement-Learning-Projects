'''
Created on Apr 10, 2018

@author: ywz
'''
import tensorflow as tf
from layers import dense


class CriticNetwork:
    
    def __init__(self, input_state, input_action, hidden_layers):
        
        assert len(hidden_layers) >= 2
        self.input_state = input_state
        self.input_action = input_action
        self.hidden_layers = hidden_layers
        
        with tf.variable_scope('critic_network'):
            self.output = self._build()
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                          tf.get_variable_scope().name)
    
    def _build(self):
        
        layer = self.input_state
        init_b = tf.constant_initializer(0.01)
        
        for i, num_unit in enumerate(self.hidden_layers):
            if i != 1:
                layer = dense(layer, num_unit, init_b=init_b, name='hidden_layer_{}'.format(i))
            else:
                layer = tf.concat([layer, self.input_action], axis=1, name='concat_action')
                layer = dense(layer, num_unit, init_b=init_b, name='hidden_layer_{}'.format(i))
        
        output = dense(layer, 1, activation=None, init_b=init_b, name='output')
        return tf.reshape(output, shape=(-1,))
    
    def get_output_layer(self):
        return self.output
    
    def get_params(self):
        return self.vars
    
    def get_value(self, sess, state):
        return sess.run(self.output, feed_dict={self.input_state: state})
    
    def get_action_value(self, sess, state, action):
        return sess.run(self.output, feed_dict={self.input_state: state,
                                                self.input_action: action})
        

if __name__ == "__main__":
    import numpy
    
    batch_size = 5
    input_dim = 10
    output_dim = 3
    hidden_layers = [20, 20]
    x = tf.placeholder(shape=(None, input_dim), dtype=tf.float32, name='input')
    a = tf.placeholder(shape=(None, input_dim), dtype=tf.float32, name='action')
    network = CriticNetwork(x, a, hidden_layers)
    
    state = numpy.random.rand(batch_size, input_dim)
    action = numpy.random.rand(batch_size, input_dim)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        value = network.get_action_value(sess, state, action)
        print(value)
                
    