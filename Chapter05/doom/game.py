'''
Created on 7 Jun 2017

@author: ywz
'''
import numpy
from vizdoom import *
from utils import cv2_resize_image

class Game:

    def __init__(self, config='basic', window_visible=True):

        self.env = DoomGame()
        self.env.load_config("./scenarios/{}.cfg".format(config))
        self.env.set_window_visible(window_visible)
        self.env.set_screen_format(ScreenFormat.GRAY8)
        self.env.init()
        
        self.env.new_episode()
        frame = self.get_current_frame()
        
        shoot = [0, 0, 1]
        left  = [1, 0, 0]
        right = [0, 1, 0]
        self.raw_actions = [shoot, left, right]
        self.actions = list(range(len(self.raw_actions)))

        self.frame_skip = 4
        self.total_reward = 0
        self.reshape_size = 120
        
        # Frame buffer
        self.buffer_size = 8
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
    
    def get_current_frame(self):
        frame = self.env.get_state().screen_buffer
        return frame
    
    def rgb_to_gray(self, im):
        if len(im) == 3:
            return numpy.dot(im, [0.299, 0.587, 0.114])
        else:
            return im
    
    def set_params(self, frame_skip=4):
        self.frame_skip = frame_skip
        self.env.new_episode()
        frame = self.get_current_frame()
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
    
    def reset(self):
        self.env.new_episode()
        frame = self.get_current_frame()
        self.total_reward = 0
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
    
    def add_frame_to_buffer(self, frame):
        self.buffer_index = self.buffer_index % self.buffer_size
        self.buffer[self.buffer_index] = frame
        self.buffer_index += 1
    
    def get_available_actions(self):
        return list(range(len(self.actions)))
    
    def get_feedback_size(self):
        return (self.reshape_size, self.reshape_size)
    
    def crop(self, frame):
        frame = cv2_resize_image(frame, 
                                 resized_shape=(self.reshape_size, self.reshape_size), 
                                 method='scale', crop_offset=0)
        return frame
    
    def get_current_feedback(self, num_frames=4):
        assert num_frames < self.buffer_size, "Frame buffer is not large enough."
        index = self.buffer_index - 1
        frames = [numpy.expand_dims(self.buffer[index - k], axis=0) for k in range(num_frames)]
        if num_frames > 1:
            return numpy.concatenate(frames, axis=0)
        else:
            return frames[0]
    
    def get_total_reward(self):
        return self.total_reward
    
    def play_action(self, action, num_frames=4):
        
        termination = 0
        a = self.raw_actions[action]
        reward = self.env.make_action(a)
        done = self.env.is_episode_finished()

        if done: 
            termination = 1
        else:
            frame = self.get_current_frame()
            self.add_frame_to_buffer(self.crop(self.rgb_to_gray(frame)))
        
        r = numpy.clip(reward, -1, 1)
        self.total_reward += reward
        
        return r, self.get_current_feedback(num_frames), termination
    
if __name__ == "__main__":
    
    import random
    from PIL import Image
    
    game = Game()
    game.set_params(frame_skip=4)
    actions = game.get_available_actions()
    print(actions)
    
    for t in range(500):
        
        action = random.choice(actions)
        reward, feedback, termination = game.play_action(action, num_frames=4)
        if termination:
            break
        
        for i in range(feedback.shape[0]):
            img = Image.fromarray(feedback[feedback.shape[0]-i-1])
            img.save('save/{}_{}.bmp'.format(t, i))
            
    