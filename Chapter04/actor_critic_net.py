'''
Created on Apr 10, 2018

@author: ywz
'''
import tensorflow as tf
from actor_network import ActorNetwork
from critic_network import CriticNetwork


class ActorCriticNet:
    
    def __init__(self, input_dim, action_dim, 
                 critic_layers, actor_layers, actor_activation, 
                 scope='ac_network'):
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.scope = scope
        
        self.x = tf.placeholder(shape=(None, input_dim), dtype=tf.float32, name='x')
        self.y = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')
        
        with tf.variable_scope(scope):
            self.actor_network = ActorNetwork(self.x, action_dim, 
                                              hidden_layers=actor_layers, 
                                              activation=actor_activation)
            
            self.critic_network = CriticNetwork(self.x, 
                                                self.actor_network.get_output_layer(),
                                                hidden_layers=critic_layers)
            
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                          tf.get_variable_scope().name)
            self._build()
    
    def _build(self):
        
        value = self.critic_network.get_output_layer()
        
        actor_loss = -tf.reduce_mean(value)
        self.actor_vars = self.actor_network.get_params()
        self.actor_grad = tf.gradients(actor_loss, self.actor_vars)
        tf.summary.scalar("actor_loss", actor_loss, collections=['actor'])
        self.actor_summary = tf.summary.merge_all('actor')
        
        critic_loss = 0.5 * tf.reduce_mean(tf.square((value - self.y)))
        self.critic_vars = self.critic_network.get_params()
        self.critic_grad = tf.gradients(critic_loss, self.critic_vars)
        tf.summary.scalar("critic_loss", critic_loss, collections=['critic'])
        self.critic_summary = tf.summary.merge_all('critic')
    
    def get_action(self, sess, state):
        return self.actor_network.get_action(sess, state)
    
    def get_value(self, sess, state):
        return self.critic_network.get_value(sess, state)
    
    def get_action_value(self, sess, state, action):
        return self.critic_network.get_action_value(sess, state, action)
    
    def get_actor_feed_dict(self, state):
        return {self.x: state}
    
    def get_critic_feed_dict(self, state, action, target):
        return {self.x: state, self.y: target, 
                self.critic_network.input_action: action}
    
    def get_clone_op(self, network, tau=0.9):
        update_ops = []
        new_vars = {v.name.replace(network.scope, ''): v for v in network.vars}
        for v in self.vars:
            u = (1 - tau) * v + tau * new_vars[v.name.replace(self.scope, '')]
            update_ops.append(tf.assign(v, u))
        return update_ops
    

if __name__ == "__main__":
    import numpy
    
    batch_size = 5
    input_dim = 10
    action_dim = 3
    hidden_layers = [20, 20]
    network = ActorCriticNet(input_dim, action_dim, 
                             hidden_layers, hidden_layers, 
                             actor_activation=tf.nn.relu)
    
    state = numpy.random.rand(batch_size, input_dim)
    action = numpy.random.rand(batch_size, action_dim)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        a = network.get_action(sess, state)
        v = network.get_value(sess, state)
        assert numpy.sum(numpy.fabs(v - network.get_action_value(sess, state, action))) > 1e-3
        assert numpy.sum(numpy.fabs(v - network.get_action_value(sess, state, a))) < 1e-8
        print("Pass")
