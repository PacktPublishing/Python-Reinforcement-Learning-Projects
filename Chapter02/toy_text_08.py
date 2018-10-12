import gym
environment = gym.make('FrozenLake-v0')
environment.reset()
environment.render()
import time
time.sleep(10)