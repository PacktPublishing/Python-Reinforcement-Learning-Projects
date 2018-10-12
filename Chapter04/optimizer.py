'''
Created on Apr 11, 2018

@author: ywz
'''
import numpy
import tensorflow as tf


class Optimizer:
    
    def __init__(self, config, ac_network, target_network, replay_memory):
        
        self.ac_network = ac_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.summary_writer = None
        self.gamma = config['gamma']
        
        if config['optimizer'] == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=config['learning_rate'], 
                                         beta1=config['rho'])
        elif config['optimizer'] == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=config['learning_rate'], 
                                             momentum=config['rho'])
        else:
            raise ValueError("Unknown optimizer")
        
        self.actor_train_op = opt.apply_gradients(zip(ac_network.actor_grad, 
                                                      ac_network.actor_vars))
        
        self.critic_train_op = opt.apply_gradients(zip(ac_network.critic_grad, 
                                                       ac_network.critic_vars))
    
    def set_summary_writer(self, summary_writer=None):
        self.summary_writer = summary_writer
        
    def sample_transitions(self, sess, batch_size):
        
        input_dim = self.ac_network.input_dim
        action_dim = self.ac_network.action_dim
        
        states = numpy.zeros((batch_size, input_dim), dtype=numpy.float32)
        new_states = numpy.zeros((batch_size, input_dim), dtype=numpy.float32)
        targets = numpy.zeros(batch_size, dtype=numpy.float32)
        actions = numpy.zeros((batch_size, action_dim), dtype=numpy.float32)
        terms = numpy.zeros(batch_size, dtype=numpy.int32)
        
        for i in range(batch_size):
            state, action, r, new_state, term = self.replay_memory.sample()
            states[i] = state
            new_states[i] = new_state
            actions[i] = action
            targets[i] = r
            terms[i] = term

        targets += self.gamma * (1 - terms) * self.target_network.get_value(sess, new_states)
        return states, actions, targets
    
    def train_one_step(self, sess, step, batch_size):
        
        states, actions, targets = self.sample_transitions(sess, batch_size)
        
        # Critic update
        feed_dict = self.ac_network.get_critic_feed_dict(states, actions, targets)
        if self.summary_writer and step % 2000 == 0:
            s, _, = sess.run([self.ac_network.critic_summary, self.critic_train_op], 
                             feed_dict=feed_dict)
            self.summary_writer.add_summary(s, step)
            self.summary_writer.flush()
        else:
            sess.run(self.critic_train_op, feed_dict=feed_dict)
        
        # Actor update 
        feed_dict = self.ac_network.get_actor_feed_dict(states)
        if self.summary_writer and step % 2000 == 0:
            s, _, = sess.run([self.ac_network.actor_summary, self.actor_train_op], 
                             feed_dict=feed_dict)
            self.summary_writer.add_summary(s, step)
            self.summary_writer.flush()
        else:
            sess.run(self.actor_train_op, feed_dict=feed_dict)
        
        
        