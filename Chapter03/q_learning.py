'''
Created on Mar 25, 2018

@author: ywz
'''
import numpy, random, os
import tensorflow as tf
from replay_memory import ReplayMemory
from optimizer import Optimizer
from q_network import QNetwork


class DQN:
    
    def __init__(self, config, game, directory, callback=None, summary_writer=None):
        
        self.game = game
        self.actions = game.get_available_actions()
        self.feedback_size = game.get_feedback_size()
        self.callback = callback
        self.summary_writer = summary_writer
        
        self.config = config
        self.batch_size = config['batch_size']
        self.n_episode = config['num_episode']
        self.capacity = config['capacity']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.num_frames = config['num_frames']
        self.num_nullops = config['num_nullops']
        self.time_between_two_copies = config['time_between_two_copies']
        self.input_scale = config['input_scale']
        self.update_interval = config['update_interval']
        self.directory = directory

        self._init_modules()
        
    def _init_modules(self):
        
        # Replay memory
        self.replay_memory = ReplayMemory(history_len=self.num_frames, 
                                          capacity=self.capacity, 
                                          batch_size=self.batch_size,
                                          input_scale=self.input_scale)
        
        input_shape = self.feedback_size + (self.num_frames,)
        # Q-network
        self.q_network = QNetwork(input_shape=input_shape, n_outputs=len(self.actions), 
                                  network_type=self.config['network_type'], scope='q_network')
        # Target network
        self.target_network = QNetwork(input_shape=input_shape, n_outputs=len(self.actions), 
                                       network_type=self.config['network_type'], scope='target_network')
        # Optimizer
        self.optimizer = Optimizer(config=self.config, 
                                   feedback_size=self.feedback_size, 
                                   q_network=self.q_network, 
                                   target_network=self.target_network, 
                                   replay_memory=self.replay_memory)
        # Ops for updating target network
        self.clone_op = self.target_network.get_clone_op(self.q_network)
        # For tensorboard
        self.t_score = tf.placeholder(dtype=tf.float32, shape=[], name='new_score')
        tf.summary.scalar("score", self.t_score, collections=['dqn'])
        self.summary_op = tf.summary.merge_all('dqn')
    
    def set_summary_writer(self, summary_writer=None):
        self.summary_writer = summary_writer
        self.optimizer.set_summary_writer(summary_writer)
    
    def choose_action(self, sess, state, epsilon_greedy):
        if numpy.random.binomial(1, epsilon_greedy) == 1:
            action = random.choice(self.actions)
        else:
            x = numpy.asarray(numpy.expand_dims(state, axis=0) / self.input_scale, dtype=numpy.float32)
            action = self.q_network.get_q_action(sess, x)[0]
        return action
    
    def play(self, action):
        r, new_state, termination = self.game.play_action(action)
        return r, new_state, termination
        
    def update_target_network(self, sess):
        sess.run(self.clone_op)
        
    def train(self, sess, saver=None):
        
        num_of_trials = -1
        for episode in range(self.n_episode):
            self.game.reset()
            frame = self.game.get_current_feedback()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
            
            for _ in range(self.config['T']):
                num_of_trials += 1
                epsilon_greedy = self.epsilon_min + \
                    max(self.epsilon_decay - num_of_trials, 0) / \
                    self.epsilon_decay * (1 - self.epsilon_min)
                print("epi {}, frame {}k: reward {}, eps {}".format(episode, 
                                                                    int(num_of_trials / 1000), 
                                                                    self.game.get_total_reward(), 
                                                                    epsilon_greedy))
                if num_of_trials % self.update_interval == 0:
                    self.optimizer.train_one_step(sess, num_of_trials, self.batch_size)
                
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, epsilon_greedy)     
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame
                
                if num_of_trials % self.time_between_two_copies == 0:
                    self.update_target_network(sess)
                    self.save(sess, saver)
                
                if self.callback:
                    self.callback()
                if termination:
                    score = self.game.get_total_reward()
                    summary_str = sess.run(self.summary_op, feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str, num_of_trials)
                    self.summary_writer.flush()
                    break
    
    def evaluate(self, sess):
        
        for episode in range(self.n_episode):
            self.game.reset()
            frame = self.game.get_current_feedback()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
            
            for _ in range(self.config['T']):
                print("episode {}, total reward {}".format(episode, 
                                                           self.game.get_total_reward()))
                
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
            
                