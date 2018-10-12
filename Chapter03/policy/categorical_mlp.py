'''
Created on 27 Sep 2017

@author: ywz
'''
import numpy
import tensorflow as tf
from mlp import MLP
from distribution.categorical import Categorical


class CategoricalMLPPolicy:
    
    def __init__(self, 
                 input_shape, 
                 output_size, 
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh):
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.locals = locals()
        
        self.distribution = Categorical(output_size)
        self.params = []
        
        with tf.variable_scope("policy"):
            # Mean network
            self.prob_mlp = MLP(input_shape=input_shape, 
                                output_size=output_size, 
                                hidden_sizes=hidden_sizes, 
                                hidden_nonlinearity=hidden_nonlinearity, 
                                output_nonlinearity=tf.nn.softmax,
                                name='prob')
            
            self.x = self.prob_mlp.get_input_layer()
            self.prob = self.prob_mlp.get_output_layer()
            self.params += self.prob_mlp.get_params()
    
    def get_locals(self):
        arguments = {argc: argv for argc, argv in self.locals.items() if argc != 'self'}
        return arguments
    
    def get_action(self, sess, observation):
        
        if observation.ndim == 1:
            observation = observation.reshape((1, observation.size))
            
        prob = sess.run(self.prob, feed_dict={self.x: observation})[0]
        idx = numpy.random.choice(range(self.output_size), p=prob)
        action = numpy.zeros((self.output_size,))
        action[idx] = 1
        
        return action, {'prob': prob}
    
    def get_actions(self, sess, observation):
        
        probs = sess.run(self.prob, feed_dict={self.x: observation})
        actions = numpy.zeros((probs.shape[0], self.output_size))
        for i, prob in enumerate(probs):
            idx = numpy.random.choice(range(self.output_size), p=prob)
            actions[i][idx] = 1
            
        return actions, {'prob': probs}
    
    def get_input(self):
        return self.x
    
    def get_dist_info(self):
        return {'prob': self.prob}
    
    def get_params(self):
        return self.params
    
    @staticmethod
    def copy(args):
        return CategoricalMLPPolicy(**args)
    
if __name__ == "__main__":
    
    input_shape = (None, 10)
    output_size = 5
    
    policy = CategoricalMLPPolicy(input_shape=input_shape,
                                  output_size=output_size)
    
    for param in policy.get_params():
        print(param)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        observation = numpy.random.rand(2, input_shape[1])
        action = policy.get_actions(sess, observation)
        print(action)
        
            
            