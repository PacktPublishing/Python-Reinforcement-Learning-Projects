import gym
environment = gym.make('HandManipulateBlock-v0')
environment.reset()
environment.render()
import time
time.sleep(10)