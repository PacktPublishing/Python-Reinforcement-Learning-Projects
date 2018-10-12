import gym
environment = gym.make('LunarLander-v2')
environment.reset()
environment.render()
import time
time.sleep(10)