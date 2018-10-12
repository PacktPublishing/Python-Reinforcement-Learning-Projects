'''
Created on 26 Sep 2017

@author: ywz
'''
import numpy
import tensorflow as tf
from mlp import MLP
from utils import iterate_minibatches


class MLPFitting:
    
    def __init__(self,
                 input_shape, 
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 learning_rate=3e-4,
                 batch_size=1000):
        
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sess = None
        
        with tf.variable_scope("mlp_fitting"):
            self.mlp = MLP(input_shape=input_shape, 
                           output_size=1, 
                           hidden_sizes=hidden_sizes, 
                           hidden_nonlinearity=hidden_nonlinearity, 
                           output_nonlinearity=None,
                           name='value')
            
            self.x = self.mlp.get_input_layer()
            self.y = tf.reshape(self.mlp.get_output_layer(), shape=(-1,))
            self.params = self.mlp.get_params()
            
            self.z = tf.placeholder(dtype=tf.float32, shape=(None,), name='z')
            loss = tf.reduce_mean(tf.square(self.z - self.y))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=self.params)
        
    def set_session(self, sess):
        self.sess = sess
        
    def train(self, paths):
        assert self.sess is not None
        obs = numpy.concatenate([path['observations'] for path in paths])
        returns = numpy.concatenate([path['returns'] for path in paths])
        if self.batch_size is not None and obs.shape[0] >= self.batch_size:
            for x, z in iterate_minibatches([obs, returns], self.batch_size, shuffle=True):
                self.sess.run(self.train_op, feed_dict={self.x: x, self.z: z})
        else:
            self.sess.run(self.train_op, feed_dict={self.x: obs, self.z: returns})
    
    def predict(self, path):
        assert self.sess is not None
        return self.sess.run(self.y, feed_dict={self.x: path['observations']})


if __name__ == "__main__":
    
    input_shape = (None, 5)
    mlp = MLPFitting(input_shape)
    
    path = {'observations': numpy.random.rand(1000, 5),
            'returns': numpy.random.rand(1000)}
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mlp.set_session(sess)
        mlp.train(paths=[path])
        print(mlp.predict(path))
    

    