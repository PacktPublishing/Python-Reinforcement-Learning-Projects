import gym
environment = gym.make('SpaceInvaders-v0')
environment.reset()
environment.render()
import time
time.sleep(10)