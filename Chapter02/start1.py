import gym
import time
environment = gym.make('CartPole-v0')
environment.reset()
for dummy in range(100):
    time.sleep(1)
    environment.render()
    environment.step(environment.action_space.sample())