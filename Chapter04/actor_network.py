'''
Created on Apr 10, 2018

@author: ywz
'''
import tensorflow as tf
from layers import dense


class ActorNetwork:
    
    def __init__(self, input_state, output_dim, hidden_layers, activation=tf.nn.relu):
        
        self.x = input_state
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        with tf.variable_scope('actor_network'):
            self.output = self._build()
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                          tf.get_variable_scope().name)
        
    def _build(self):
        
        layer = self.x
        init_b = tf.constant_initializer(0.01)
        
        for i, num_unit in enumerate(self.hidden_layers):
            layer = dense(layer, num_unit, init_b=init_b, name='hidden_layer_{}'.format(i))
            
        output = dense(layer, self.output_dim, activation=self.activation, init_b=init_b, name='output')
        return output
    
    def get_output_layer(self):
        return self.output
    
    def get_params(self):
        return self.vars
    
    def get_action(self, sess, state):
        return sess.run(self.output, feed_dict={self.x: state})
    

if __name__ == "__main__":
    import numpy
    
    batch_size = 5
    input_dim = 10
    output_dim = 3
    hidden_layers = [20, 20]
    x = tf.placeholder(shape=(None, input_dim), dtype=tf.float32, name='input')
    network = ActorNetwork(x, output_dim, hidden_layers)
    
    state = numpy.random.rand(batch_size, input_dim)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        action = network.get_action(sess, state)
        print(action)
    
    