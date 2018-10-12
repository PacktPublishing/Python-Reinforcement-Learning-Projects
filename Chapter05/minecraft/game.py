'''
Created on Jul 10, 2018

@author: ywz
'''
import gym
import gym_minecraft
import minecraft_py
import numpy, time
from utils import cv2_resize_image


class Game:

    def __init__(self, name='MinecraftBasic-v0', discrete_movement=False):
        
        self.env = gym.make(name)
        if discrete_movement:
            self.env.init(start_minecraft=True, allowDiscreteMovement=["move", "turn"])
        else:
            self.env.init(start_minecraft=True, allowContinuousMovement=["move", "turn"])
        self.actions = list(range(self.env.action_space.n))
        frame = self.env.reset()
        
        self.frame_skip = 1
        self.total_reward = 0
        self.crop_size = 84
        
        # Frame buffer
        self.buffer_size = 8
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame
    
    def rgb_to_gray(self, im):
        return numpy.dot(im, [0.2126, 0.7152, 0.0722])
    
    def set_params(self, crop_size=84, frame_skip=4):
        
        self.crop_size = crop_size
        self.frame_skip = frame_skip
        
        frame = self.env.reset()
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame
    
    def reset(self):
        frame = self.env.reset()
        self.total_reward = 0
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame
    
    def add_frame_to_buffer(self, frame):
        self.buffer_index = self.buffer_index % self.buffer_size
        self.buffer[self.buffer_index] = frame
        self.buffer_index += 1
    
    def get_available_actions(self):
        return list(range(len(self.actions)))
    
    def get_feedback_size(self):
        return (self.crop_size, self.crop_size)
    
    def crop(self, frame):
        feedback = cv2_resize_image(frame, 
                                    resized_shape=(self.crop_size, self.crop_size), 
                                    method='scale', crop_offset=0)
        return feedback
    
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
        
        reward = 0
        termination = 0
        for i in range(self.frame_skip):
            a = self.actions[action]
            frame, r, done, _ = self.env.step(a)
            reward += r
            if i == self.frame_skip - 2: 
                self.last_frame = frame
            if done: 
                termination = 1
        self.add_frame_to_buffer(self.crop(numpy.maximum(self.rgb_to_gray(frame), self.rgb_to_gray(self.last_frame))))
        
        r = numpy.clip(reward, -1, 1)
        self.total_reward += reward
        
        return r, self.get_current_feedback(num_frames), termination
    
    def draw(self):
        time.sleep(1 / 120.0)
        self.env.render(mode='human')
