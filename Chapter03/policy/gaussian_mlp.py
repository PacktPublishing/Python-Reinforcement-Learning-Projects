'''
Created on 5 Sep 2017

@author: ywz
'''
import numpy
import tensorflow as tf
from mlp import MLP
from distribution.diagonal_gaussian import DiagonalGaussian


class GaussianMLPPolicy:
    
    def __init__(self, 
                 input_shape, 
                 output_size, 
                 hidden_sizes=(32, 32),
                 learn_std=True, 
                 init_std=1.0, 
                 adaptive_std=False,
                 std_hidden_sizes=(32, 32), 
                 min_std=1e-6, 
                 std_parametrization='exp',
                 hidden_nonlinearity=tf.nn.tanh, 
                 output_nonlinearity=None,
                 std_hidden_nonlinearity=tf.nn.tanh):
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.std_parametrization = std_parametrization
        self.locals = locals()
        
        self.distribution = DiagonalGaussian(output_size)
        self.params = []
        
        with tf.variable_scope("policy"):
            # Mean network
            self.mean_mlp = MLP(input_shape=input_shape, 
                                output_size=output_size, 
                                hidden_sizes=hidden_sizes, 
                                hidden_nonlinearity=hidden_nonlinearity, 
                                output_nonlinearity=output_nonlinearity,
                                name='mean')
            
            self.x = self.mean_mlp.get_input_layer()
            self.mean = self.mean_mlp.get_output_layer()
            self.params += self.mean_mlp.get_params()
            
            # Var network
            if adaptive_std:
                self.var_mlp = MLP(input_shape=input_shape, 
                                   output_size=output_size, 
                                   hidden_sizes=std_hidden_sizes, 
                                   hidden_nonlinearity=std_hidden_nonlinearity, 
                                   output_nonlinearity=None,
                                   input_layer=self.x,
                                   name='var')
                self.log_var = self.var_mlp.get_output_layer()
                self.params += self.var_mlp.get_params()
            else:
                if std_parametrization == 'exp':
                    init_std_param = numpy.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = numpy.log(numpy.exp(init_std) - 1)
                else:
                    raise NotImplementedError
                
                with tf.variable_scope('var'):
                    self.log_var = tf.get_variable(name='v', 
                                                   shape=(output_size,), 
                                                   dtype=tf.float32, 
                                                   initializer=tf.constant_initializer(init_std_param, dtype=tf.float32), 
                                                   trainable=learn_std)
                    self.log_var = tf.tile(tf.reshape(self.log_var, shape=(-1, output_size)), 
                                           multiples=[tf.shape(self.x)[0], 1])
                    
                    self.params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            
            if std_parametrization == 'exp':
                min_std_param = numpy.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = numpy.log(numpy.exp(min_std) - 1)
            else:
                raise NotImplementedError
                
            self.log_var = tf.maximum(self.log_var, min_std_param)
            if self.std_parametrization == 'softplus':
                self.log_var = tf.log(tf.log(1. + tf.exp(self.log_var)))
    
    def get_locals(self):
        arguments = {argc: argv for argc, argv in self.locals.items() if argc != 'self'}
        return arguments
    
    def get_action(self, sess, observation):
        if observation.ndim == 1:
            observation = observation.reshape((1, observation.size))
        mean, log_var = sess.run([self.mean, self.log_var], feed_dict={self.x: observation})
        mean, log_var = mean[0], log_var[0]
        action = mean + numpy.random.normal(size=mean.shape) * numpy.exp(log_var)
        return action, {'mean': mean, 'log_var': log_var}
    
    def get_actions(self, sess, observation):
        mean, log_var = sess.run([self.mean, self.log_var], feed_dict={self.x: observation})
        action = mean + numpy.random.normal(size=mean.shape) * numpy.exp(log_var)
        return action, {'mean': mean, 'log_var': log_var}
    
    def get_input(self):
        return self.x
    
    def get_dist_info(self):
        return {'mean': self.mean, 'log_var': self.log_var}
    
    def get_params(self):
        return self.params
    
    @staticmethod
    def copy(args):
        return GaussianMLPPolicy(**args)


if __name__ == "__main__":
    
    input_shape = (None, 10)
    output_size = 5
    
    policy = GaussianMLPPolicy(input_shape=input_shape,
                               output_size=output_size,
                               learn_std=True,
                               adaptive_std=False)
    
    with tf.variable_scope("new"):
        new_policy = type(policy).copy(policy.get_locals())
    
    for param in policy.get_params():
        print(param)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        observation = numpy.random.rand(1, input_shape[1])
        action = policy.get_action(sess, observation)
        print(action)
    
    
    
                