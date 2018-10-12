'''
Created on 18 Sep 2017

@author: ywz
'''
import gym, numpy
from gym import spaces


class Simulator:
    
    # Supported tasks: 
    # v1: Reacher, HalfCheetah, Hopper, Swimmer, Walker2d, Ant, Humanoid
    # v0: CartPole, Acrobot, Pendulum
    def __init__(self, task='Swimmer'):
        
        self.task = task
        try:
            self.env = gym.make('{}-v1'.format(task))
        except:
            self.env = gym.make('{}-v2'.format(task))
        self.env.reset()
        
        if type(self.env.action_space) == spaces.Box:        
            assert len(self.env.action_space.shape) == 1
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
        elif type(self.env.action_space) == spaces.Discrete:
            self.action_dim = self.env.action_space.n
            self.action_type = 'discrete'
        else:
            raise NotImplementedError
        
        assert len(self.env.observation_space.shape) == 1
        self.obsevation_dim = self.env.observation_space.shape[0]
        self.total_reward = 0
    
    def reset(self):
        self.total_reward = 0
        return self.env.reset()
    
    def play(self, action):
        
        termination = 0
        if self.action_type == 'continuous':
            observation, reward, done, _ = self.env.step(action)
        elif self.action_type == 'discrete':
            observation, reward, done, _ = self.env.step(numpy.argmax(action))
        
        if done: termination = 1
        self.total_reward += reward
        
        return observation, reward, termination
    
    def render(self):
        self.env.render()
    
    def get_total_reward(self):
        return self.total_reward
    
    
if __name__ == "__main__":

    agent = Simulator(task='Swimmer')
    
    for _ in range(10):
        observation = agent.reset()
        while True:
            action = numpy.random.uniform(low=-1.0, high=1.0, size=(agent.action_dim,))
            observation, reward, termination = agent.play(action)
            
            print("Observation: {}".format(observation))
            print("Action: {}".format(action))
            print("Reward: {}".format(reward))
            print("Termination: {}".format(termination))
            
            if termination:
                break
            agent.render()
            
    
    
    