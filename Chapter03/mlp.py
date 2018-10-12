'''
Created on 5 Sep 2017

@author: ywz
'''
import tensorflow as tf
from layer import linear


class MLP:
    
    def __init__(self, input_shape, output_size, hidden_sizes=(32, 32), 
                 hidden_nonlinearity=tf.nn.relu, output_nonlinearity=tf.nn.tanh,
                 input_layer=None, name='mlp'):
        
        self.input_shape = input_shape
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.name = name
        
        if input_layer is None:
            self.x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='mlp_input')
        else:
            self.x = input_layer
            
        self.build()
    
    def build(self):
        
        with tf.variable_scope(self.name):
            layer = self.x
            for i, hidden_size in enumerate(self.hidden_sizes):
                layer = linear(layer, hidden_size, activation=self.hidden_nonlinearity, 
                               init_b=tf.constant_initializer(0.0), name='hidden_layer_{}'.format(i))
                
            self.y = linear(layer, self.output_size, activation=self.output_nonlinearity, 
                            init_b=tf.constant_initializer(0.0), name='output_layer')
            
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        
    def get_params(self):
        return self.params
    
    def get_input_layer(self):
        return self.x
    
    def get_output_layer(self):
        return self.y
        

if __name__ == "__main__":
    
    import numpy
    
    input_shape = (None, 10)
    output_size = 5
    mlp = MLP(input_shape=input_shape, output_size=output_size)
    print(mlp.get_params())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("log/", sess.graph_def)
        
        x = numpy.random.rand(1, input_shape[1])
        y = sess.run(mlp.get_output_layer(), feed_dict={mlp.get_input_layer(): x})
        print(y)
    
    
    
    
        
        