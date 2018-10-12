'''
Created on 5 Sep 2017

@author: ywz
'''
import tensorflow as tf
from mlp import MLP


class DeterministicMLPPolicy:
    
    def __init__(self, 
                 input_shape, 
                 output_size, 
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu, 
                 output_nonlinearity=tf.nn.tanh):
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.locals = locals()
        
        with tf.variable_scope("policy"):
            self.mlp = MLP(input_shape=input_shape, 
                           output_size=output_size, 
                           hidden_sizes=hidden_sizes, 
                           hidden_nonlinearity=hidden_nonlinearity, 
                           output_nonlinearity=output_nonlinearity)
        
        self.x = self.mlp.get_input_layer()
        self.y = self.mlp.get_output_layer()
    
    def get_locals(self):
        arguments = {argc: argv for argc, argv in self.locals.items() if argc != 'self'}
        return arguments
    
    def get_action(self, sess, observation):
        if observation.ndim == 1:
            observation = observation.reshape((1, observation.size))
        output = sess.run(self.y, feed_dict={self.x: observation})
        return output[0]
    
    def get_actions(self, sess, observation):
        return sess.run(self.y, feed_dict={self.x: observation})
    
    def get_params(self):
        return self.mlp.get_params()
    
    @staticmethod
    def copy(args):
        return DeterministicMLPPolicy(**args)
    
    
    