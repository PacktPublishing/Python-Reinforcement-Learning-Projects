'''
Created on 29 May 2017

@author: ywz
'''
import numpy, os
import tensorflow as tf
from ff_policy import FFPolicy
from lstm_policy import LSTMPolicy
from utils import update_target_graph, create_optimizer


class A3C:
    
    def __init__(self, system, directory, param, agent_index=0, callback=None):
        
        self.system = system
        self.actions = system.get_available_actions()
        self.directory = directory
        self.callback = callback
        self.feedback_size = system.get_feedback_size()
        self.agent_index = agent_index
        
        self.set_params(param)
        self.init_network()
        
        self.summary_writer = None
        self.num_episodes = 500000
        self.eval_counter = 0
        self.eval_frequency = 4 * 10 ** 5
    
    def set_params(self, param):
        
        self.gamma = param['gamma']
        self.num_frames = param['num_frames']
        self.T = param['iteration_num']
        self.async_update_interval = param['async_update_interval']
        self.network_type = param['network_type']
        
        self.learning_rate = param['learning_rate']
        self.rho = param['rho'] 
        self.update_method = param['update_method']
        self.max_iter_num = param['max_iter_num']
        self.rmsprop_epsilon = param['rmsprop_epsilon']
        self.input_scale = param['input_scale']
        
        self.use_lstm = True
        self.shared_optimizer = False
        self.anneal_learning_rate = False
    
    def set_summary_writer(self, writer=None):
        self.summary_writer = writer
    
    def init_network(self):
        
        input_shape = self.feedback_size + (self.num_frames,)
        worker_device = "/job:worker/task:{}/cpu:0".format(self.agent_index)
        
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if self.use_lstm is False:
                    self.shared_network = FFPolicy(input_shape, len(self.actions), self.network_type)
                else:
                    self.shared_network = LSTMPolicy(input_shape, len(self.actions), self.network_type)
                    
                self.global_step = tf.get_variable("global_step", shape=[], 
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False, dtype=tf.int32)
                self.best_score = tf.get_variable("best_score", shape=[], 
                                                   initializer=tf.constant_initializer(-1e2, dtype=tf.float32),
                                                   trainable=False, dtype=tf.float32)
                
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                if self.use_lstm is False:
                    self.network = FFPolicy(input_shape, len(self.actions), self.network_type)
                else:
                    self.network = LSTMPolicy(input_shape, len(self.actions), self.network_type)
                # Sync params
                self.update_local_ops = update_target_graph(self.shared_network.vars, self.network.vars)
                # Learning rate
                self.lr = tf.get_variable(name='lr', shape=[], 
                                          initializer=tf.constant_initializer(self.learning_rate),
                                          trainable=False, dtype=tf.float32)
                self.t_lr = tf.placeholder(dtype=tf.float32, shape=[], name='new_lr')
                self.assign_lr_op = tf.assign(self.lr, self.t_lr)
                # Best score
                self.t_score = tf.placeholder(dtype=tf.float32, shape=[], name='new_score')
                self.assign_best_score_op = tf.assign(self.best_score, self.t_score)
                # Build gradient_op
                self.increase_step = self.global_step.assign_add(1)
                gradients = self.network.build_gradient_op(clip_grad=40.0)
                # Additional summaries
                tf.summary.scalar("learning_rate", self.lr, collections=['a3c'])
                tf.summary.scalar("score", self.t_score, collections=['a3c'])
                tf.summary.scalar("best_score", self.best_score, collections=['a3c'])
                self.summary_op = tf.summary.merge_all('a3c')
        
        if self.shared_optimizer:
            with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
                with tf.variable_scope("global"):
                    optimizer = create_optimizer(self.update_method, self.lr, self.rho, self.rmsprop_epsilon)
                    self.train_op = optimizer.apply_gradients(zip(gradients, self.shared_network.vars))
        else:
            with tf.device(worker_device):
                with tf.variable_scope('local'):
                    optimizer = create_optimizer(self.update_method, self.lr, self.rho, self.rmsprop_epsilon)
                    self.train_op = optimizer.apply_gradients(zip(gradients, self.shared_network.vars))
    
    def n_step_q_learning(self, sess, replay_memory, cell):
        
        batch_size = len(replay_memory)
        w, h = self.system.get_feedback_size()
        states = numpy.zeros((batch_size, self.num_frames, w, h), dtype=numpy.float32)
        rewards = numpy.zeros(batch_size, dtype=numpy.float32)
        advantages = numpy.zeros(batch_size, dtype=numpy.float32)
        actions = numpy.zeros((batch_size, len(self.actions)), dtype=numpy.float32)
        
        for i in reversed(range(batch_size)):
            state, action, r, new_state, value, termination = replay_memory[i]
            states[i] = state
            actions[i][action] = 1
            
            if termination != 0:
                rewards[i] = r
            else:
                if i == batch_size - 1:
                    rewards[i] = r + self.gamma * self.Q_value(sess, new_state, cell)
                else:
                    rewards[i] = r + self.gamma * rewards[i+1]
            advantages[i] = rewards[i] - value
        
        return states, actions, rewards, advantages
    
    def Q_value(self, sess, state, cell):
        state = numpy.expand_dims(state, axis=0)
        return self.network.run_value(sess, state, cell)[0]

    def choose_action(self, sess, state, cell):
        state = numpy.expand_dims(state, axis=0)
        output = self.network.run_policy_and_value(sess, state, cell)
        probs = output[0][0]
        value = output[1][0]
        cell = None if len(output) <= 2 else output[2]
        return numpy.random.choice(range(len(probs)), p=probs), value, cell
    
    def play(self, action):
        r, new_state, termination = self.system.play_action(action, self.num_frames)
        return r, new_state, termination
    
    def anneal_lr(self, iter_num=None):
        if self.anneal_learning_rate and iter_num is not None:
            lr = max(self.learning_rate * (self.max_iter_num - iter_num) / self.max_iter_num, 1e-6)
        else:
            lr = self.learning_rate
        return lr
    
    def train(self, sess, states, actions, rewards, advantages, init_cell, iter_num):
        
        lr = self.anneal_lr(iter_num)
        feed_dict = self.network.get_feed_dict(states, actions, rewards, advantages, init_cell)
        sess.run(self.assign_lr_op, feed_dict={self.t_lr: lr})
        
        step = int((iter_num - self.async_update_interval + 1) / self.async_update_interval)
        if self.summary_writer and step % 10 == 0:
            summary_str, _, step = sess.run([self.network.summary_op, self.train_op, self.global_step], 
                                            feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, step)
            self.summary_writer.flush()
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            
    def run(self, sess, saver=None):
        
        num_of_trials = -1
        for episode in range(self.num_episodes):
            
            self.system.reset()
            cell = self.network.run_initial_state(sess)
            state = self.system.get_current_feedback(self.num_frames)
            state = numpy.asarray(state / self.input_scale, dtype=numpy.float32)
            replay_memory = []
            
            for _ in range(self.T):
                num_of_trials += 1
                global_step = sess.run(self.increase_step)
                if len(replay_memory) == 0:
                    init_cell = cell
                    sess.run(self.update_local_ops)
                
                action, value, cell = self.choose_action(sess, state, cell)
                r, new_state, termination = self.play(action)
                new_state = numpy.asarray(new_state / self.input_scale, dtype=numpy.float32)
                replay = (state, action, r, new_state, value, termination)
                replay_memory.append(replay)
                state = new_state
                
                print(("agent {:2d}, episode {:5d}, frame {:4d}k, g_step {:4d}k, " + \
                       "reward {:5d}, action {:2d}, q_value {:4f}").format(self.agent_index, episode,
                                                                           int(num_of_trials / 1000),
                                                                           int(global_step / 1000),
                                                                           int(self.system.get_total_reward()), 
                                                                           action, value))

                if len(replay_memory) == self.async_update_interval or termination:
                    states, actions, rewards, advantages = self.n_step_q_learning(sess, replay_memory, cell)
                    self.train(sess, states, actions, rewards, advantages, init_cell, num_of_trials)
                    replay_memory = []

                if global_step % 40000 == 0:
                    self.save(sess, saver)
                if self.callback:
                    self.callback()
                if termination:
                    score = self.system.get_total_reward()
                    summary_str = sess.run(self.summary_op, feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str, global_step)
                    self.summary_writer.flush()
                    break

            if global_step - self.eval_counter > self.eval_frequency:
                self.evaluate(sess, n_episode=10, saver=saver)
                self.eval_counter = global_step
    
    def evaluate(self, sess, n_episode=10, saver=None, verbose=False):
        
        average_score = 0
        lost_life_as_terminal = self.system.lost_life_as_terminal
        self.system.lost_life_as_terminal = False
        sess.run(self.update_local_ops)
        
        for episode in range(n_episode):
            self.system.reset()
            cell = self.network.run_initial_state(sess)
            state = self.system.get_current_feedback(self.num_frames)
            state = numpy.asarray(state / self.input_scale, dtype=numpy.float32)
            
            for step in range(self.T):
                action, _, cell = self.choose_action(sess, state, cell)
                _, new_state, termination = self.play(action)
                new_state = numpy.asarray(new_state / self.input_scale, dtype=numpy.float32)
                state = new_state
                
                if verbose:
                    print("episode {:4d}, frame {:4d}, total reward {:5f}".format(episode, step,
                                                                                  self.system.get_total_reward()))
                if self.callback:
                    self.callback()
                if termination:
                    break
            average_score += self.system.get_total_reward()
        
        self.system.lost_life_as_terminal = lost_life_as_terminal
        average_score /= n_episode
        best_score = sess.run(self.best_score)
        if saver and average_score > best_score:
            sess.run(self.assign_best_score_op, feed_dict={self.t_score: average_score})
            self.save(sess, saver, model_name='best_a3c_model.ckpt')
    
    def save(self, sess, saver, model_name='a3c_model.ckpt'):
        if saver:
            try:
                checkpoint_path = os.path.join(self.directory, model_name)
                saver.save(sess, checkpoint_path)
            except:
                pass
    
    def load(self, sess, saver, model_name='a3c_model.ckpt'):
        if saver:
            try:
                checkpoint_path = os.path.join(self.directory, model_name)
                saver.restore(sess, checkpoint_path)
            except:
                pass
            
    