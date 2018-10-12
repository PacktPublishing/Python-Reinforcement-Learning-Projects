'''
Created on May 16, 2016

@author: a0096049
'''

import numpy, pygame
from demo.utils import Color, calculateIntersectPoint


class Object:
    
    def __init__(self, x, y, r, game):
        
        self.x = x
        self.y = y
        self.r = r
        self.game = game
        
    def get_position(self):
        return self.x, self.y
    
    def get_radius(self):
        return self.r
    
    def set_position(self, x, y):
        self.x = x
        self.y = y
        
    def draw(self):
        pass
    
class Food(Object):
    
    def __init__(self, x, y, radius, t, game):
        
        super().__init__(x, y, radius, game)
        self.type = t
        self.life = numpy.random.randint(1000, 5000)
        
    def decrease_life(self):
        self.life -= 1
        return self.life == 0

    def draw(self, found=False):
        
        if found == False:
            if self.type == "bad":
                pygame.draw.circle(self.game.DISPLAYSURF, Color.RED, (self.x, self.y), self.r)
            else:
                pygame.draw.circle(self.game.DISPLAYSURF, Color.GREEN, (self.x, self.y), self.r)
        else:
            pygame.draw.circle(self.game.DISPLAYSURF, Color.BLUE, (self.x, self.y), self.r)
    
class Wall:
    
    def __init__(self, start, end, game, width=2):
        
        self.start = start
        self.end = end
        self.game = game
        self.width = width
        
    def draw(self):
        pygame.draw.line(self.game.DISPLAYSURF, Color.WHITE, self.start, self.end, self.width)
    
    def collide(self, p1, p2):
        
        point = calculateIntersectPoint(p1, p2, self.start, self.end)
        if point is None:
            return None
        else:
            return (int(point[0]), int(point[1]))
        
    
        
    
