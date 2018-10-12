'''
Created on 7 Jun 2017

@author: ywz
'''
from vizdoom import *
import random
import time

def main():

    game = DoomGame()
    game.load_config("./scenarios/basic.cfg")
    game.init()
    
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]
    
    episodes = 10
    for _ in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            
            print(img.shape)
            print(misc)
            
            reward = game.make_action(random.choice(actions))
            print("\treward: {}".format(reward))
            time.sleep(0.05)
        print("Result: {}".format(game.get_total_reward()))
        time.sleep(2)
        
if __name__ == "__main__":
    main()
    
    