import gym
environment = gym.make('CartPole-v0')
environment.reset()
environment.render()
import time
time.sleep(10)