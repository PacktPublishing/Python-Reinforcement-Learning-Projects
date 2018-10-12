'''
Created on Apr 11, 2018

@author: ywz
'''
import gym
import numpy
import tensorflow as tf


class Task:
    
    def __init__(self, name):
        
        assert name in ['CartPole-v0', 'MountainCar-v0', 
                        'Pendulum-v0', 'Acrobot-v1']
        self.name = name
        self.task = gym.make(name)
        self.last_state = self.reset()
    
    def reset(self):
        state = self.task.reset()
        self.total_reward = 0
        return state
        
    def play_action(self, action):
        
        if self.name not in ['Pendulum-v0', 'MountainCarContinuous-v0']:
            action = numpy.fmax(action, 0)
            action = action / numpy.sum(action)
            action = numpy.random.choice(range(len(action)), p=action)
        else:
            low = self.task.env.action_space.low
            high = self.task.env.action_space.high
            action = numpy.fmin(numpy.fmax(action, low), high)
            
        state, reward, done, _ = self.task.step(action)
        self.total_reward += reward
        termination = 1 if done else 0
        
        return reward, state, termination
    
    def get_total_reward(self):
        return self.total_reward
    
    def get_action_dim(self):
        if self.name not in ['Pendulum-v0', 'MountainCarContinuous-v0']:
            return self.task.env.action_space.n
        else:
            return self.task.env.action_space.shape[0]
    
    def get_state_dim(self):
        return self.last_state.shape[0]
    
    def get_activation_fn(self):
        if self.name not in ['Pendulum-v0', 'MountainCarContinuous-v0']:
            return tf.nn.softmax
        else:
            return None
    
    def render(self):
        self.task.render()
    