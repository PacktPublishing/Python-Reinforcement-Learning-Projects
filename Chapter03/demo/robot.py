'''
Created on May 18, 2016

@author: a0096049
'''
import numpy, pygame
from demo.object import Object
from demo.utils import Color


class Sensor:
    
    def __init__(self, center, angle, length, owner):
        
        self.owner = owner
        self.center = center
        self.length = length
        self.angle = angle
        self.orientation = 0
        
        self.end_point = None
        self.collide_point = None
    
    def set_center(self, x, y):
        self.center = (x, y)
    
    def set_angle(self, angle):
        self.angle = angle
        
    def set_orientation(self, orient):
        self.orientation = orient
    
    def set_end_point(self, x, y):
        self.end_point = (x, y)
    
    def restore_end_point(self):
        
        angle = self.angle + self.orientation
        dx = int(self.length * numpy.sin(numpy.deg2rad(angle)))
        dy = int(-self.length * numpy.cos(numpy.deg2rad(angle)))
        self.end_point = (self.center[0]+dx, self.center[1]+dy)
    
    def detect_wall(self, walls):
        
        self.collide_point = None
        minimum_distance = self.length * 2
        detected_wall = None
        self.restore_end_point()
        
        for wall in walls:
            point = wall.collide(self.center, self.end_point)
            if point is not None:
                distance = numpy.sqrt((self.center[0] - point[0]) ** 2 + (self.center[1] - point[1]) ** 2)
                if distance < minimum_distance:
                    minimum_distance = distance
                    self.collide_point = point
                    detected_wall = wall
        
        return detected_wall, minimum_distance
                
    def draw(self):
        
        if self.collide_point is None:
            self.restore_end_point()
            pygame.draw.line(self.owner.game.DISPLAYSURF, Color.GRAY, self.center, self.end_point, 1)
        else:
            pygame.draw.line(self.owner.game.DISPLAYSURF, Color.GRAY, self.center, self.collide_point, 1)
        
    
class Robot(Object):
    
    def __init__(self, x, y, radius, sensor_num, sensor_length, game):
        
        super().__init__(x, y, radius, game)
        
        self.sensors = [Sensor([x, y], 0, sensor_length, self) for _ in range(sensor_num)]
        self.sensor_length = sensor_length
        self.sensor_angle = 15
        self.sensor_angle_start = -int(sensor_num / 2) * self.sensor_angle
        for i in range(sensor_num):
            self.sensors[i].set_angle(self.sensor_angle_start + i * self.sensor_angle)
        self.sensor_angle_end = self.sensor_angle_start + (sensor_num-1) * self.sensor_angle
        
        self.orientation = 0
        self.step_size = self.r / 2
        self.actions = [self.sensor_angle*2, -self.sensor_angle, 0, self.sensor_angle, -self.sensor_angle*2]
        
        self.total_reward = 0
        
    def set_reward(self, r):
        self.total_reward = r
    
    def add_reward(self, r):
        self.total_reward += r
    
    def _valid_angle(self, angle):
        if angle > 180:
            angle -= 360
        elif angle <= -180:
            angle += 360
        return angle
    
    def play_action(self, action):
        
        if action >= len(self.actions):
            return
        else:
            self.orientation = self._valid_angle(self.orientation + self.actions[action])
            for sensor in self.sensors:
                sensor.set_orientation(self.orientation)
            
    def move_one_step(self):
        
        dx = int(self.step_size * numpy.sin(numpy.deg2rad(self.orientation)))
        dy = int(-self.step_size * numpy.cos(numpy.deg2rad(self.orientation)))
        x, y = self.game.get_valid_position(self.x, self.y, self.x+dx, self.y+dy)
        self.set_position(x, y)
        for sensor in self.sensors:
            sensor.set_center(x, y)
    
    def explore_food(self):

        foods = self.game.get_foods()
        found_foods = []
        
        for food in foods:
            x, y = food.get_position()
            dx = x - self.x
            dy = y - self.y
            distance = numpy.sqrt(dx ** 2 + dy ** 2)
            if distance >= self.sensor_length + food.r:
                continue
            
            angle = numpy.rad2deg(numpy.arctan2(dx, -dy))
            angle = self._valid_angle(angle - self.orientation)
            rectified_angle = numpy.rad2deg(numpy.arctan2(food.r, distance))
            if angle >= self.sensor_angle_start-rectified_angle and angle <= self.sensor_angle_end+rectified_angle:
                found_foods.append(food)
        
        return found_foods
    
    def detect_wall(self):
        
        walls = self.game.get_walls()
        detected_walls = []
        distances = []
        
        for sensor in self.sensors:
            detected_wall, distance = sensor.detect_wall(walls)
            detected_walls.append(detected_wall)
            distances.append(distance)
        
        return detected_walls, distances
    
    def sensor_feedback(self):
        
        thres = 2
        foods_found_by_robot = self.explore_food()
        feedbacks = numpy.ones((len(self.sensors), 3), dtype="float32") * thres
        
        for food in foods_found_by_robot:
            
            x, y = food.get_position()
            dx = x - self.x
            dy = y - self.y
            distance = numpy.sqrt(dx ** 2 + dy ** 2)
            angle = numpy.rad2deg(numpy.arctan2(dx, -dy))
            angle = self._valid_angle(angle - self.orientation)
            rectified_angle = numpy.rad2deg(numpy.arctan2(food.r, distance))
            
            distance /= self.sensor_length
            for i, sensor in enumerate(self.sensors):
                if numpy.abs(angle - sensor.angle) <= rectified_angle:
                    if food.type == "bad":
                        feedbacks[i, 1] = min(feedbacks[i, 1], distance)
                    else:
                        feedbacks[i, 0] = min(feedbacks[i, 0], distance)
        
        detected_walls, distances = self.detect_wall()
        for i in range(len(self.sensors)):
            feedbacks[i, 2] = distances[i] / self.sensor_length
            if feedbacks[i, 0] > feedbacks[i, 2]:
                feedbacks[i, 0] = thres
            if feedbacks[i, 1] > feedbacks[i, 2]:
                feedbacks[i, 1] = thres
        
        return feedbacks.flatten(), foods_found_by_robot, detected_walls
    
    def eat_nearby_food(self, foods):
        
        reward = 0
        eaten_foods = []
        
        for food in foods:
            
            x, y = food.get_position()
            dx = x - self.x
            dy = y - self.y
            distance = numpy.sqrt(dx ** 2 + dy ** 2)
            if distance < self.r + food.r:
                r = -1 if food.type == "bad" else +1
                reward += r
                eaten_foods.append(food)
        
        self.add_reward(reward)
        return eaten_foods, reward
    
    def get_feedback_size(self):
        return 3 * len(self.sensors)
    
    def get_total_reward(self):
        return self.total_reward
    
    def get_actions(self):
        return list(range(len(self.actions)))
    
    def draw(self):
        
        pygame.draw.circle(self.game.DISPLAYSURF, Color.BLUE, (self.x, self.y), self.r)
        for sensor in self.sensors:
            sensor.draw()
    