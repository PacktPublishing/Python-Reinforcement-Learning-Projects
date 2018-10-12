'''
Created on 18 Sep 2017

@author: ywz
'''
import numpy, math
import tensorflow as tf


class TRPO:
    
    def __init__(self, policy, optimizer, step_size):
        
        self.policy = policy
        self.optimizer = optimizer
        self.step_size = step_size
        
        self.x = self.policy.get_input()
        self.action_dim = self.policy.output_size
        self.dist = self.policy.distribution
        
        self.build_formula()
        
    def build_formula(self):
        
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self.action_dim), name='action')
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None,), name='action')
        
        dist_vars = self.policy.get_dist_info()
        old_dist_vars = {k: tf.placeholder(tf.float32, shape=[None]+list(shape), name='old_dist_{}'.format(k))
                         for k, shape in self.dist.specs()}
        old_dist_vars_list = [old_dist_vars[k] for k in self.dist.keys()]
        
        kl = self.dist.kl_tf(old_dist_vars, dist_vars)
        lr = self.dist.likelihood_ratio_tf(self.action, old_dist_vars, dist_vars)
        mean_kl = tf.reduce_mean(kl)
        loss = -tf.reduce_mean(lr * self.advantage)
        
        self.inputs_tensors = [self.x, self.action, self.advantage] + old_dist_vars_list
        self.optimizer.build(loss=loss, 
                             leq_constraint=(mean_kl, self.step_size), 
                             params=self.policy.get_params(), 
                             inputs=self.inputs_tensors)
        # Add summaries
        tf.summary.scalar("loss", loss, collections=['trpo'])
        tf.summary.scalar("mean_kl", mean_kl, collections=['trpo'])
        self.summary_op = tf.summary.merge_all('trpo')
        
    def optimize_policy(self, sess, samples, logger=None, subsample_rate=0.5):
        
        if subsample_rate < 1.0:
            n = len(samples['rewards'])
            idx = numpy.random.choice(n, int(math.floor(n * subsample_rate)), replace=False)
            obs = samples['observations'][idx]
            actions = samples['actions'][idx]
            advantages = samples['advantages'][idx]
            dist_vars = [samples['infos'][k][idx] for k in self.dist.keys()]
        else:
            obs = samples['observations']
            actions = samples['actions']
            advantages = samples['advantages']
            dist_vars = [samples['infos'][k] for k in self.dist.keys()]
        
        inputs = [obs, actions, advantages] + dist_vars
        self.optimizer.optimize(sess, input_vals=inputs)
        
        if logger:
            feed_dict = dict(list(zip(self.inputs_tensors, inputs)))
            summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
            logger.add_summary(summary_str)
        

if __name__ == "__main__":
    from policy.gaussian_mlp import GaussianMLPPolicy
    from optimizer import ConjugateOptimizer
    
    input_shape = (None, 10)
    output_size = 5
    
    policy = GaussianMLPPolicy(input_shape=input_shape,
                               output_size=output_size,
                               learn_std=True,
                               adaptive_std=False)
    optimizer = ConjugateOptimizer()
    
    trpo = TRPO(policy, optimizer, step_size=0.01)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        samples = {}
        samples['observations'] = numpy.random.rand(10, input_shape[1])
        samples['actions'] = numpy.random.rand(10, output_size)
        samples['advantages'] = numpy.random.rand(10)
        samples['infos'] = {'mean': numpy.random.rand(10, output_size),
                            'log_var': numpy.random.rand(10, output_size)}
        
        trpo.optimize_policy(sess, samples, subsample_rate=1.0)
        print("Finished.")
            
            