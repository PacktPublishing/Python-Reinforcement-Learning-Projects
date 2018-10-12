'''
Created on Jan 24, 2017

@author: ywz
'''
import pickle
from utils import log_uniform

class Parameter:
    
    def __init__(self, lr, directory=None):
        
        self.directory = directory
            
        if isinstance(lr, tuple):
            assert len(lr) == 2
            assert lr[0] < lr[1]
            self.learning_rate = log_uniform(lr[0], lr[1])
        else:
            self.learning_rate = lr
        
        self.gamma = 0.99
        self.num_history_frames = 4
        self.iteration_num = 100000
        self.async_update_interval = 5
            
        self.rho = 0.99
        self.rmsprop_epsilon = 1e-6
        self.update_method = 'rmsprop'
        self.clip_delta = 0
        self.max_iter_num = 10 ** 8
        self.network_type = 'cnn'
        self.input_scale = 255.0
        
    def get(self):
        
        param = {}
        param['directory'] = self.directory
        param['learning_rate'] = self.learning_rate
        
        param['gamma'] = self.gamma
        param['num_frames'] = self.num_history_frames
        param['iteration_num'] = self.iteration_num
        param['async_update_interval'] = self.async_update_interval
        
        param['rho'] = self.rho
        param['rmsprop_epsilon'] = self.rmsprop_epsilon
        param['update_method'] = self.update_method
        param['clip_delta'] = self.clip_delta
        param['max_iter_num'] = self.max_iter_num
        param['network_type'] = self.network_type
        param['input_scale'] = self.input_scale
        
        return param
    
    def __str__(self):
        param = self.get()
        strs = ["{}: {}".format(key, value) for key, value in param.items()]
        return "\n".join(strs)
    
    def save(self, filename):
        assert self.directory is not None
        filename = '{}/{}'.format(self.directory, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.get(), f)
            