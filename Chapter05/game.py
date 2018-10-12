'''
Created on Jan 14, 2017

@author: ywz
'''
import gym, numpy, time
from utils import cv2_resize_image, preprocess_image


class Game:

    def __init__(self, name, lost_life_as_terminal=False, take_maximum_of_two_frames=False):
        
        if take_maximum_of_two_frames is False:
            self.mode = 'Deterministic'
        else:
            self.mode = 'NoFrameskip'
            
        name = ''.join([s.capitalize() for s in name.split('_')])
        self.ale = gym.make('{}{}-v4'.format(name, self.mode))
        frame = self.ale.reset()
        self.lost_life_as_terminal = lost_life_as_terminal
        self.lives = 0
        self.actions = list(range(self.ale.action_space.n))
        
        self.frame_skip = 4
        self.total_reward = 0
        self.crop_size = 84
        self.crop_offset = 8
        
        # Frame buffer
        self.buffer_size = 8
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        # Overlapping frames, maximum of two frames
        self.last_frame = frame
    
    def rgb_to_gray(self, im):
        return numpy.dot(im, [0.2126, 0.7152, 0.0722])
    
    def set_params(self, crop_size=84, crop_offset=8, frame_skip=4, 
                   lost_life_as_terminal=False, take_maximum_of_two_frames=False):
        
        self.crop_size = crop_size
        self.crop_offset = crop_offset
        self.frame_skip = frame_skip
        self.lost_life_as_terminal = lost_life_as_terminal
        self.mode = 'NoFrameskip' if take_maximum_of_two_frames else 'Deterministic'
        
        frame = self.ale.reset()
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame
    
    def reset(self):
        frame = self.ale.reset()
        self.total_reward = 0
        self.buffer_index = 0
        self.lives = 0
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
    
    def crop(self, frame, mode='new'):
        if mode == 'old':
            return preprocess_image(frame, (100, 84), self.crop_size, 'down')
        elif mode == 'new':
            feedback = cv2_resize_image(frame, 
                                        resized_shape=(self.crop_size, self.crop_size), 
                                        method='crop', crop_offset=self.crop_offset)
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
    
    def _lost_life(self, info):
        if self.lost_life_as_terminal:
            lives = info['ale.lives']
            if lives >= self.lives:
                self.lives = lives
                return False
            else:
                return True
        else:
            return False
    
    def play_action(self, action, num_frames=4):
        
        if self.mode == 'Deterministic':
            termination = 0
            a = self.actions[action]
            frame, reward, done, info = self.ale.step(a)
            if done or self._lost_life(info): termination = 1
            self.add_frame_to_buffer(self.crop(self.rgb_to_gray(frame)))
        elif self.mode == 'NoFrameskip':
            reward = 0
            termination = 0
            for i in range(self.frame_skip):
                a = self.actions[action]
                frame, r, done, info = self.ale.step(a)
                reward += r
                if i == self.frame_skip - 2: self.last_frame = frame
                if done or self._lost_life(info): termination = 1
            self.add_frame_to_buffer(self.crop(numpy.maximum(self.rgb_to_gray(frame), self.rgb_to_gray(self.last_frame))))
        else:
            raise
        
        r = numpy.clip(reward, -1, 1)
        self.total_reward += reward
        
        return r, self.get_current_feedback(num_frames), termination
    
    def draw(self):
        time.sleep(1 / 120.0)
        self.ale.render()
    
if __name__ == "__main__":
    
    import pygame
    import random
    from PIL import Image
    
    pygame.init()
    game = Game('breakout')
    game.set_params(frame_skip=3, take_maximum_of_two_frames=False)
    actions = game.get_available_actions()
    print(actions)
    
    for t in range(100):
        
        action = random.choice(actions)
        reward, feedback, termination = game.play_action(action, num_frames=4)

        for i in range(feedback.shape[0]):
            img = Image.fromarray(feedback[feedback.shape[0]-i-1])
            img.save('save/{}_{}.bmp'.format(t, i))
    
    '''
    keys = {}
    keys[pygame.K_1] = 0
    keys[pygame.K_2] = 1
    keys[pygame.K_3] = 2
    keys[pygame.K_4] = 3
    keys[pygame.K_5] = 4
    keys[pygame.K_6] = 5
    keys[pygame.K_7] = 6
    keys[pygame.K_8] = 7
    keys[pygame.K_9] = 8
    keys[pygame.K_0] = 9
    keys_down = {key: 0 for key in keys.keys()}
    
    while True:
        time.sleep(1.0 / 120)
        game.draw()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if keys.get(event.key, None) is not None:
                    keys_down[event.key] = 1
            elif event.type == pygame.KEYUP:
                if keys.get(event.key, None) is not None:
                    keys_down[event.key] = 0
        
        action = 0
        for key, value in keys_down.items():
            if value == 1:
                action = keys[key]
                break
        if action != -1:
            reward, feedback, termination = game.play_action(action, num_frames=4)
    '''
            
            