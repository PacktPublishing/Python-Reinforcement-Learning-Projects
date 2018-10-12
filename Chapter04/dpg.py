'''
Created on Apr 12, 2018

@author: ywz
'''
import numpy, os
import tensorflow as tf
from replay_memory import ReplayMemory
from optimizer import Optimizer
from actor_critic_net import ActorCriticNet


class DPG:
    
    def __init__(self, config, task, directory, callback=None, summary_writer=None):
        
        self.task = task
        self.directory = directory
        self.callback = callback
        self.summary_writer = summary_writer
        
        self.config = config
        self.batch_size = config['batch_size']
        self.n_episode = config['num_episode']
        self.capacity = config['capacity']
        self.history_len = config['history_len']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.time_between_two_copies = config['time_between_two_copies']
        self.update_interval = config['update_interval']
        self.tau = config['tau']
        
        self.action_dim = task.get_action_dim()
        self.state_dim = task.get_state_dim() * self.history_len
        self.critic_layers = [50, 50]
        self.actor_layers = [50, 50]
        self.actor_activation = task.get_activation_fn()
        
        self._init_modules()
        
    def _init_modules(self):
        
        # Replay memory
        self.replay_memory = ReplayMemory(history_len=self.history_len, 
                                          capacity=self.capacity)
        # Actor critic network
        self.ac_network = ActorCriticNet(input_dim=self.state_dim, 
                                         action_dim=self.action_dim, 
                                         critic_layers=self.critic_layers, 
                                         actor_layers=self.actor_layers, 
                                         actor_activation=self.actor_activation,
                                         scope='ac_network')
        # Target network
        self.target_network = ActorCriticNet(input_dim=self.state_dim, 
                                             action_dim=self.action_dim, 
                                             critic_layers=self.critic_layers, 
                                             actor_layers=self.actor_layers, 
                                             actor_activation=self.actor_activation,
                                             scope='target_network')
        # Optimizer
        self.optimizer = Optimizer(config=self.config, 
                                   ac_network=self.ac_network, 
                                   target_network=self.target_network, 
                                   replay_memory=self.replay_memory)
        # Ops for updating target network
        self.clone_op = self.target_network.get_clone_op(self.ac_network, tau=self.tau)
        # For tensorboard
        self.t_score = tf.placeholder(dtype=tf.float32, shape=[], name='new_score')
        tf.summary.scalar("score", self.t_score, collections=['dpg'])
        self.summary_op = tf.summary.merge_all('dpg')
    
    def set_summary_writer(self, summary_writer=None):
        self.summary_writer = summary_writer
        self.optimizer.set_summary_writer(summary_writer)
        
    def choose_action(self, sess, state, epsilon=0.1):
        x = numpy.asarray(numpy.expand_dims(state, axis=0), dtype=numpy.float32)
        action = self.ac_network.get_action(sess, x)[0]
        return action + epsilon * numpy.random.randn(len(action))
    
    def play(self, action):
        r, new_state, termination = self.task.play_action(action)
        return r, new_state, termination
        
    def update_target_network(self, sess):
        sess.run(self.clone_op)
        
    def train(self, sess, saver=None):
        
        num_of_trials = -1
        for episode in range(self.n_episode):
            frame = self.task.reset()
            for _ in range(self.history_len+1):
                self.replay_memory.add(frame, 0, 0, 0)
            
            for _ in range(self.config['T']):
                num_of_trials += 1
                epsilon = self.epsilon_min + \
                          max(self.epsilon_decay - num_of_trials, 0) / \
                          self.epsilon_decay * (1 - self.epsilon_min)
                print("epi {}, frame {}k, epsilon {}".format(episode, num_of_trials // 1000, epsilon))
                if num_of_trials % self.update_interval == 0:
                    self.optimizer.train_one_step(sess, num_of_trials, self.batch_size)
                
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, epsilon)     
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                if num_of_trials % self.time_between_two_copies == 0:
                    self.update_target_network(sess)
                    self.save(sess, saver)
                
                if self.callback:
                    self.callback()
                if termination:
                    score = self.task.get_total_reward()
                    summary_str = sess.run(self.summary_op, feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str, num_of_trials)
                    self.summary_writer.flush()
                    break
    
    def evaluate(self, sess):
        
        for episode in range(self.n_episode):
            frame = self.task.reset()
            for _ in range(self.history_len+1):
                self.replay_memory.add(frame, 0, 0, 0)
            
            for _ in range(self.config['T']):
                print("episode {}, total reward {}".format(episode, 
                                                           self.task.get_total_reward()))
                
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, self.epsilon_min)     
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame

                if self.callback:
                    self.callback()
                    if termination:
                        break
    
    def save(self, sess, saver, model_name='model.ckpt'):
        if saver:
            try:
                checkpoint_path = os.path.join(self.directory, model_name)
                saver.save(sess, checkpoint_path)
            except:
                pass
    
    def load(self, sess, saver, model_name='model.ckpt'):
        if saver:
            try:
                checkpoint_path = os.path.join(self.directory, model_name)
                saver.restore(sess, checkpoint_path)
            except:
                pass
        