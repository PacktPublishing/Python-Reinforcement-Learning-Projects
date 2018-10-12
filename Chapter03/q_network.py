'''
Created on Mar 25, 2018

@author: ywz
'''
import tensorflow as tf
from layers import conv2d, dense


class QNetwork:
    
    def __init__(self, input_shape=(84, 84, 4), n_outputs=4, 
                 network_type='cnn', scope='q_network'):
        
        self.width = input_shape[0]
        self.height = input_shape[1]
        self.channel = input_shape[2]
        self.n_outputs = n_outputs
        self.network_type = network_type
        self.scope = scope
        
        # Frame images
        self.x = tf.placeholder(dtype=tf.float32, 
                                shape=(None, self.channel, self.width, self.height))
        # Estimates of Q-value
        self.y = tf.placeholder(dtype=tf.float32, shape=(None,))
        # Selected actions
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,))
        
        with tf.variable_scope(scope):
            self.build()
            self.build_loss()
        
    def build(self):
        
        self.net = {}
        self.net['input'] = tf.transpose(self.x, perm=(0, 2, 3, 1))
            
        if self.network_type == 'cnn':
            self.net['conv1'] = conv2d(self.net['input'], 32, kernel=(8, 8), stride=(4, 4), 
                                       init_b=tf.constant_initializer(0.01), name='conv1')
            self.net['conv2'] = conv2d(self.net['input'], 64, kernel=(4, 4), stride=(2, 2), 
                                       init_b=tf.constant_initializer(0.01), name='conv2')
            self.net['conv3'] = conv2d(self.net['input'], 64, kernel=(3, 3), stride=(1, 1), 
                                       init_b=tf.constant_initializer(0.01), name='conv3')
            self.net['feature'] = dense(self.net['conv2'], 512, 
                                        init_b=tf.constant_initializer(0.01), name='fc1')
        elif self.network_type == 'cnn_nips':
            self.net['conv1'] = conv2d(self.net['input'], 16, kernel=(8, 8), stride=(4, 4), 
                                       init_b=tf.constant_initializer(0.01), name='conv1')
            self.net['conv2'] = conv2d(self.net['conv1'], 32, kernel=(4, 4), stride=(2, 2), 
                                       init_b=tf.constant_initializer(0.01), name='conv2')
            self.net['feature'] = dense(self.net['conv2'], 256, 
                                        init_b=tf.constant_initializer(0.01), name='fc1')
        elif self.network_type == 'mlp':
            self.net['fc1'] = dense(self.net['input'], 50, 
                                    init_b=tf.constant_initializer(0.0), name='fc1')
            self.net['feature'] = dense(self.net['fc1'], 50, 
                                        init_b=tf.constant_initializer(0.0), name='fc2')
        else:
            raise NotImplementedError('Unknown network type: {}'.format(self.network_type))
            
        self.net['values'] = dense(self.net['feature'], self.n_outputs, activation=None,
                                   init_b=tf.constant_initializer(0.0), name='values')
        
        self.net['q_value'] = tf.reduce_max(self.net['values'], axis=1, name='q_value')
        self.net['q_action'] = tf.argmax(self.net['values'], axis=1, 
                                         name='q_action', output_type=tf.int32)
        
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                      tf.get_variable_scope().name)
    
    def build_loss(self):
        
        indices = tf.transpose(tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=0))
        value = tf.gather_nd(self.net['values'], indices, name='action_value')
        
        self.loss = 0.5 * tf.reduce_mean(tf.square((value - self.y)))
        self.gradient = tf.gradients(self.loss, self.vars)
        
        tf.summary.scalar("loss", self.loss, collections=['q_network'])
        self.summary_op = tf.summary.merge_all('q_network')
        
    def get_q_value(self, sess, state):
        return sess.run(self.net['q_value'], feed_dict={self.x: state})
    
    def get_q_action(self, sess, state):
        return sess.run(self.net['q_action'], feed_dict={self.x: state})
    
    def get_feed_dict(self, states, actions, values):
        return {self.x: states, self.a: actions, self.y: values}
                
    def get_clone_op(self, network):
        new_vars = {v.name.replace(network.scope, ''): v for v in network.vars}
        return [tf.assign(v, new_vars[v.name.replace(self.scope, '')]) for v in self.vars]
    

if __name__ == "__main__":
    import numpy
    
    num_actions = 4
    batch_size = 5
    network = QNetwork(n_outputs=num_actions)
    
    state = numpy.random.rand(batch_size, 4, 84, 84)
    values = numpy.random.rand(batch_size)
    actions = numpy.random.randint(num_actions, size=batch_size)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        q_values = sess.run(network.net['values'], feed_dict={network.x: state})
        q_value = network.get_q_value(sess, state)
        q_action = network.get_q_action(sess, state)
        
        print(q_values)
        print(q_value)
        print(q_action)
    
