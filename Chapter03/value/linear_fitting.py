'''
Created on 20 Sep 2017

@author: ywz
'''
import numpy


class LinearFitting:
    
    def __init__(self):
        self.beta = None
        self.sess = None
    
    def set_session(self, sess):
        self.sess = sess
    
    def feature(self, path):
        o = numpy.clip(path['observations'], -10, 10)
        l = len(path["rewards"])
        al = numpy.arange(l).reshape(-1, 1) / 100.0
        return numpy.concatenate([o, o ** 2, al, al ** 2, al ** 3, numpy.ones((l, 1))], axis=1)
    
    def train(self, paths):
        
        features = numpy.concatenate([self.feature(path) for path in paths])
        returns = numpy.concatenate([path['returns'] for path in paths])

        reg_coeff = 1e-5
        for _ in range(5):
            self.beta = numpy.linalg.lstsq(features.T.dot(features) + 
                                           reg_coeff * numpy.identity(features.shape[1]), 
                                           features.T.dot(returns))[0]
            if not numpy.any(numpy.isnan(self.beta)):
                break
            reg_coeff *= 10
    
    def predict(self, path):
        if self.beta is None:
            return numpy.zeros((len(path['rewards'],)))
        else:
            return self.feature(path).dot(self.beta)
        
            