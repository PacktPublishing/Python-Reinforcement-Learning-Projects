'''
Created on 22 Sep 2017

@author: ywz
'''
import os
import argparse
import tensorflow as tf
from simulator import Simulator
from sampler import Sampler
from policy.gaussian_mlp import GaussianMLPPolicy
from policy.categorical_mlp import CategoricalMLPPolicy


def test(task, 
         num_episodes=10,
         policy_network_hidden_sizes=(32, 32),
         policy_adaptive_std=False):
    
    directory = 'log/{}/'.format(task)
    simulator = Simulator(task=task)
    
    input_shape = (None, simulator.obsevation_dim)
    output_size = simulator.action_dim
    
    if simulator.action_type == 'continuous':
        policy_network = GaussianMLPPolicy(input_shape=input_shape,
                                           output_size=output_size,
                                           hidden_sizes=policy_network_hidden_sizes,
                                           adaptive_std=policy_adaptive_std,
                                           std_hidden_sizes=policy_network_hidden_sizes)
    elif simulator.action_type == 'discrete':
        policy_network = CategoricalMLPPolicy(input_shape=input_shape,
                                              output_size=output_size,
                                              hidden_sizes=policy_network_hidden_sizes)
    
    sampler = Sampler(simulator, policy_network)
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(directory, '{}.ckpt'.format(task))
        saver.restore(sess, checkpoint_path)
        
        for i in range(num_episodes):
            path = sampler.rollout(sess, max_path_length=1000, render=True)
            print("epsiode {}, reward {}".format(i, path['total_reward']))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-t', '--task', default='Swimmer', 
                        type=str, help='Tasks: Swimmer, Walker2d, Reacher, HalfCheetah, Hopper, Ant, Humanoid')
    args = parser.parse_args()
    
    test(task=args.task, policy_network_hidden_sizes=(32, 32))
    
    
            
    