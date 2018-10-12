'''
Created on 18 Sep 2017

@author: ywz
'''
import numpy
import tensorflow as tf


class DiagonalGaussian:
    
    def __init__(self, dim):
        self.dim = dim
        
    def specs(self):
        return [("mean", (self.dim,)), ("log_var", (self.dim,))]
    
    def keys(self):
        return ["mean", "log_var"]
    
    def kl_numpy(self, old_dist, new_dist):
        
        old_means = old_dist["mean"]
        old_log_stds = old_dist["log_var"]
        new_means = new_dist["mean"]
        new_log_stds = new_dist["log_var"]

        old_std = numpy.exp(old_log_stds)
        new_std = numpy.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = numpy.square(old_means - new_means) + numpy.square(old_std) - numpy.square(new_std)
        denominator = 2 * numpy.square(new_std) + 1e-8
        
        return numpy.sum(numerator / denominator + new_log_stds - old_log_stds, axis=-1)
    
    def kl_tf(self, old_dist, new_dist):
        
        old_means = old_dist["mean"]
        old_log_stds = old_dist["log_var"]
        new_means = new_dist["mean"]
        new_log_stds = new_dist["log_var"]

        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = tf.square(old_means - new_means) + tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        
        return tf.reduce_sum(numerator / denominator + new_log_stds - old_log_stds, axis=-1)
    
    def likelihood_ratio_tf(self, x, old_dist, new_dist):
        
        new = self.log_likelihood_tf(x, new_dist)
        old = self.log_likelihood_tf(x, old_dist)
        
        return tf.exp(new - old)

    def log_likelihood_tf(self, x, dist):
        
        means = dist["mean"]
        log_stds = dist["log_var"]
        zs = (x - means) / tf.exp(log_stds)
        
        return - tf.reduce_sum(log_stds, axis=-1) - \
               0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
               0.5 * self.dim * numpy.log(2 * numpy.pi)
    
        
    