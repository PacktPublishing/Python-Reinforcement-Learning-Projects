'''
Created on May 16, 2016

@author: a0096049
'''

import numpy
import pygame
from pygame.locals import *
from demo.object import Food, Wall
from demo.robot import Robot
from demo.utils import Color
        
        
class Game:
    
    def __init__(self, w, h, DISPLAYSURF):
        
        self.w = w
        self.h = h
        self.DISPLAYSURF = DISPLAYSURF
        self.fpsClock = pygame.time.Clock()
        self.fps = 60
        
        self.obj_radius = 10
        self.robot_sensor_num = 9 
        self.robot_sensor_length = 80
        self.number_of_food = 40
        self.collide_with_wall_penalty = -0.1
        
        self.num_of_rounds = 0
        self.check_terminate = True
        self.lost_life_as_terminal = False
        
        self.foods = {}
        self.walls = []
        self.robot = Robot(x=0, y=0, radius=self.obj_radius, sensor_num=self.robot_sensor_num, 
                           sensor_length=self.robot_sensor_length, game=self)
        self.feedback_size = self.robot.get_feedback_size()
        
    def init_player_position(self):
        
        x = numpy.random.randint(0, self.w)
        y = numpy.random.randint(0, self.h)
        self.robot.set_position(x, y)
        self.robot.set_reward(r=0)
        return x, y
    
    def generate_food(self, bad_food_prob=0.5):
        
        while True:      
              
            x = numpy.random.randint(0, self.w)
            y = numpy.random.randint(0, self.h)
            radius = self.obj_radius * 2
            
            if self.collide_with_walls(x-radius, y-radius, x+radius, y+radius):
                continue
            if self.collide_with_walls(x-radius, y+radius, x+radius, y-radius):
                continue
            
            t = "bad" if numpy.random.binomial(1, bad_food_prob) == 1 else "good"
            if self.foods.get((x, y), None) is None:
                self.foods[(x, y)] = Food(x=x, y=y, radius=self.obj_radius, t=t, game=self)
                break
    
    def remove_food(self, food):
        
        x, y = food.get_position()
        if self.foods.get((x, y), None) is not None:
            del self.foods[(x, y)]
    
    def init_walls(self):
        
        self.walls = []
        
        start = (0, 0)
        end = (self.w, 0)
        self.walls.append(Wall(start=start, end=end, game=self, width=1))
        
        start = (0, self.h-1)
        end = (self.w-1, self.h-1)
        self.walls.append(Wall(start=start, end=end, game=self, width=1))
        
        start = (0, 0)
        end = (0, self.h-1)
        self.walls.append(Wall(start=start, end=end, game=self, width=1))
        
        start = (self.w-1, 0)
        end = (self.w-1, self.h-1)
        self.walls.append(Wall(start=start, end=end, game=self, width=1))
        
        start = (int(self.w/2), int(self.h/3))
        end = (int(self.w/2), int(self.h*2/3))
        self.walls.append(Wall(start=start, end=end, game=self))
        
        start = (int(self.w/5), int(self.h/5))
        end = (int(self.w/5), int(self.h*4/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        '''
        start = (int(self.w/5), int(self.h/5))
        end = (int(self.w/5)+int(self.w/7), int(self.h/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        
        start = (int(self.w/5), int(self.h*4/5))
        end = (int(self.w/5)+int(self.w/7), int(self.h*4/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        '''
        start = (int(self.w*4/5), int(self.h/5))
        end = (int(self.w*4/5), int(self.h*4/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        '''
        start = (int(self.w*4/5)-int(self.w/7), int(self.h/5))
        end = (int(self.w*4/5), int(self.h/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        
        start = (int(self.w*4/5)-int(self.w/7), int(self.h*4/5))
        end = (int(self.w*4/5), int(self.h*4/5))
        self.walls.append(Wall(start=start, end=end, game=self))
        '''
        
    def reset(self):
        
        self.foods = {}
        self.init_player_position()
        for _ in range(self.number_of_food):
            self.generate_food(bad_food_prob=0.5)
        self.init_walls()
        self.num_of_rounds = 0
    
    def collide_with_walls(self, x0, y0, x1, y1):
        
        flag = False
        for wall in self.walls:
            point = wall.collide((x0, y0), (x1, y1))
            if point is not None:
                flag = True
                break
        return flag
    
    def get_valid_position(self, x0, y0, x1, y1):
        
        
        dx = x1 - x0
        dy = y1 - y0
        angle = numpy.rad2deg(numpy.arctan2(dx, -dy))
        if angle >= -45 and angle <= 45:
            flag = self.collide_with_walls(x0, y0, x1-self.obj_radius, y1-self.obj_radius) or self.collide_with_walls(x0, y0, x1+self.obj_radius, y1-self.obj_radius)
        elif angle > 45 and angle <= 135:
            flag = self.collide_with_walls(x0, y0, x1+self.obj_radius, y1-self.obj_radius) or self.collide_with_walls(x0, y0, x1+self.obj_radius, y1+self.obj_radius)
        elif angle > 135 or angle < -135:
            flag = self.collide_with_walls(x0, y0, x1-self.obj_radius, y1+self.obj_radius) or self.collide_with_walls(x0, y0, x1+self.obj_radius, y1+self.obj_radius)
        else:
            flag = self.collide_with_walls(x0, y0, x1-self.obj_radius, y1-self.obj_radius) or self.collide_with_walls(x0, y0, x1-self.obj_radius, y1+self.obj_radius)
         
        if flag:
            x = x0
            y = y0
        else:
            x = x1
            y = y1
            
        if x < 0: x = 0
        if x >= self.w: x = self.w - 1
        if y < 0: y = 0
        if y >= self.h: y = self.h - 1
        return x, y
    
    def get_feedback_size(self):
        return (self.feedback_size, 1)
    
    def play_action(self, action, num_frames=1):
        
        assert num_frames == 1
        self.num_of_rounds += 1
        x0, y0 = self.robot.get_position()
        self.robot.play_action(action)
        self.robot.move_one_step()
        x1, y1 = self.robot.get_position()
        feedbacks, foods_found_by_robot, _ = self.robot.sensor_feedback()
        eaten_foods, reward = self.robot.eat_nearby_food(foods_found_by_robot)
        # Collide with the walls, reward = collide_with_wall_penalty
        if x1 == x0 and y1 == y0:
            reward = self.collide_with_wall_penalty
        
        for food in eaten_foods:
            self.remove_food(food)
            self.generate_food(bad_food_prob=0.5)
        dead_foods = []
        for food in self.foods.values():
            if food.decrease_life():
                dead_foods.append(food)
        for food in dead_foods:
            self.remove_food(food)
            self.generate_food(bad_food_prob=0.5)
        
        termination = 0
        if self.check_terminate:
            if self.get_total_reward() < -10 or self.num_of_rounds > 5001:
                termination = 1
        
        return reward, feedbacks.reshape((1, len(feedbacks), 1)), termination
    
    def move_robot(self):
        self.robot.move_one_step()
    
    def get_foods(self):
        return list(self.foods.values())
    
    def get_walls(self):
        return self.walls
    
    def get_total_reward(self):
        return self.robot.get_total_reward()
    
    def get_available_actions(self):
        return self.robot.get_actions()
    
    def get_current_feedback(self, num_frames=1):
        assert num_frames == 1
        feedbacks, _, _ = self.robot.sensor_feedback()
        return feedbacks.reshape((1, len(feedbacks), 1))
    
    def get_number_of_foods(self):
        return len(list(self.foods.keys()))
    
    def increase_fps(self):
        self.fps = min(300, self.fps+10)
        
    def decrease_fps(self):
        self.fps = max(1, self.fps-10)
    
    def draw(self, *params):

        # Draw objects
        self.DISPLAYSURF.fill(Color.BACKGROUND)
        for food in self.foods.values():
            food.draw()
        for wall in self.walls:
            wall.draw()
        self.robot.draw()

        pygame.display.update()
        self.fpsClock.tick(self.fps)
            
        
