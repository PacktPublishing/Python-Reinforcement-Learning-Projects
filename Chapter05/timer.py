'''
Created on Jan 20, 2017

@author: ywz
'''
import time

class Timer:
    
    def __init__(self):
        self.total_time = 0
        self.current_time = 0
        self.name = ''
        
    def reset(self):
        self.total_time = 0
        self.current_time = 0
    
    def set_name(self, name):
        self.name = name
        
    def begin(self):
        self.current_time = time.time()
        
    def end(self):
        self.total_time += time.time() - self.current_time
        
    def total_time(self):
        return self.total_time
    
    def print(self):
        print('{} took {}s'.format(self.name, self.total_time))
        