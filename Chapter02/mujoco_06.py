import gym
environment = gym.make('Humanoid-v2')
environment.reset()
environment.render()
import time
time.sleep(10)
