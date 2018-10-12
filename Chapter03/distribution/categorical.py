'''
Created on 27 Sep 2017

@author: ywz
'''
import numpy
import tensorflow as tf


class Categorical:
    
    def __init__(self, dim):
        self.dim = dim
        
    def specs(self):
        return [("prob", (self.dim,))]
    
    def keys(self):
        return ["prob"]
    
    def kl_numpy(self, old_dist, new_dist):
        
        old_prob = old_dist["prob"]
        new_prob = new_dist["prob"]
        
        return numpy.sum(old_prob * (numpy.log(old_prob + 1e-8) - numpy.log(new_prob + 1e-8)), axis=-1)
    
    def kl_tf(self, old_dist, new_dist):

        old_prob = old_dist["prob"]
        new_prob = new_dist["prob"]

        return tf.reduce_sum(old_prob * (tf.log(old_prob + 1e-8) - tf.log(new_prob + 1e-8)), axis=-1)
    
    def likelihood_ratio_tf(self, x, old_dist, new_dist):
        
        old_prob = old_dist["prob"]
        new_prob = new_dist["prob"]

        return (tf.reduce_sum(new_prob * x, axis=-1) + 1e-8) / \
               (tf.reduce_sum(old_prob * x, axis=-1) + 1e-8)
    
        