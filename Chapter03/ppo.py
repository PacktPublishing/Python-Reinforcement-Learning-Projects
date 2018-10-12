'''
Created on 26 Sep 2017

@author: ywz
'''
import tensorflow as tf
from utils import iterate_minibatches

# Proximal Policy Optimization Algorithms
class PPO:
    
    def __init__(self, 
                 policy, 
                 batch_size=1000, 
                 learning_rate=1e-3, 
                 epsilon=0.2):
        
        self.policy = policy
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        
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
        
        lr = self.dist.likelihood_ratio_tf(self.action, old_dist_vars, dist_vars)
        first_term = lr * self.advantage
        second_term = tf.clip_by_value(lr, 1 - self.epsilon, 1 + self.epsilon) * self.advantage
        loss = -tf.reduce_mean(tf.minimum(first_term, second_term))
        
        self.inputs_tensors = [self.x, self.action, self.advantage] + old_dist_vars_list
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, 
                                                                            var_list=self.policy.get_params())
        # Add summaries
        tf.summary.scalar("loss", loss, collections=['ppo'])
        self.summary_op = tf.summary.merge_all('ppo')
        
    def optimize_policy(self, sess, samples, logger=None, **args):
        
        obs = samples['observations']
        actions = samples['actions']
        advantages = samples['advantages']
        dist_vars = [samples['infos'][k] for k in self.dist.keys()]
        
        inputs = [obs, actions, advantages] + dist_vars
        feed_dict = dict(list(zip(self.inputs_tensors, inputs)))
        if self.batch_size is not None and obs.shape[0] >= self.batch_size:
            for vs in iterate_minibatches(inputs, self.batch_size, shuffle=True):
                sess.run(self.train_op, feed_dict=dict(list(zip(self.inputs_tensors, vs))))
        else:
            sess.run(self.train_op, feed_dict=feed_dict)

        if logger:
            summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
            logger.add_summary(summary_str)
            
            