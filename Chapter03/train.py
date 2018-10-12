'''
Created on 21 Sep 2017

@author: ywz
'''
import argparse
import numpy, os
import tensorflow as tf
from trpo import TRPO
from ppo import PPO
from simulator import Simulator
from optimizer import ConjugateOptimizer
from sampler import Sampler, ParallelSampler
from policy.gaussian_mlp import GaussianMLPPolicy
from policy.categorical_mlp import CategoricalMLPPolicy
from value.linear_fitting import LinearFitting
from value.mlp_fitting import MLPFitting
from logger import Logger


class Trainer:
    
    def __init__(self,
                 task,
                 num_epsiodes=500,
                 discount_factor=0.995,
                 gae_lambda = 1.0,
                 trpo_step_size=0.01,
                 policy_network_hidden_sizes=(64, 64),
                 policy_learn_std=True,
                 policy_adaptive_std=False,
                 cg_iters=10,
                 cg_damping=1e-5,
                 cg_backtrack_ratio=0.8,
                 cg_max_backtracks=10,
                 sampler_thread_num=8,
                 sampler_max_samples=50000,
                 sampler_max_path_length=1000,
                 sampler_center_advantage=True):
        
        self.task = task
        self.discount = discount_factor
        self.gae_lambda = gae_lambda
        self.sampler_max_samples = sampler_max_samples
        self.sampler_max_path_length = sampler_max_path_length
        self.sampler_center_advantage = sampler_center_advantage
        self.subsample_rate = 0.8
        self.fitting_mode = 'linear'
        self.use_trpo = True
        
        self.num_episodes = num_epsiodes
        self.directory = 'log/{}/'.format(task)
        
        self.simulator = Simulator(task=task)
        input_shape = (None, self.simulator.obsevation_dim)
        output_size = self.simulator.action_dim
        
        if self.fitting_mode == 'linear':
            self.value_network = LinearFitting()
        elif self.fitting_mode == 'mlp':
            self.value_network = MLPFitting(input_shape, hidden_sizes=(32, 32))
        else:
            raise NotImplementedError
        
        if self.simulator.action_type == 'continuous':
            self.policy_network = GaussianMLPPolicy(input_shape=input_shape,
                                                    output_size=output_size,
                                                    hidden_sizes=policy_network_hidden_sizes,
                                                    learn_std=policy_learn_std,
                                                    adaptive_std=policy_adaptive_std,
                                                    std_hidden_sizes=policy_network_hidden_sizes)
        elif self.simulator.action_type == 'discrete':
            self.policy_network = CategoricalMLPPolicy(input_shape=input_shape,
                                                       output_size=output_size,
                                                       hidden_sizes=policy_network_hidden_sizes)
        
        self.optimizer = ConjugateOptimizer(cg_iters=cg_iters,
                                            reg_coeff=cg_damping,
                                            backtrack_ratio=cg_backtrack_ratio,
                                            max_backtracks=cg_max_backtracks)
        
        self.sampler = Sampler(self.simulator, self.policy_network)
        self.parallel_sampler = ParallelSampler(self.sampler, 
                                                thread_num=sampler_thread_num, 
                                                max_path_length=self.sampler_max_path_length,
                                                render=False)
        
        if self.use_trpo:
            self.trpo = TRPO(self.policy_network, self.optimizer, trpo_step_size)
        else:
            self.trpo = PPO(self.policy_network)
        
        # Additional summaries
        self.average_reward = tf.placeholder(dtype=tf.float32, shape=[])
        tf.summary.scalar("reward", self.average_reward, collections=['trainer'])
        self.summary_op = tf.summary.merge_all('trainer')
        
    def run(self):
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            logger = Logger(sess=sess, directory=self.directory)
            self.value_network.set_session(sess)
            sess.run(tf.global_variables_initializer())
            
            for i in range(self.num_episodes):
                logger.set_step(step=i)
                # Generate simulation paths
                self.parallel_sampler.update_policy_params(sess)
                paths = self.parallel_sampler.generate_paths(max_num_samples=self.sampler_max_samples)
                paths = self.parallel_sampler.truncate_paths(paths, max_num_samples=self.sampler_max_samples)
                # Compute the average reward of the sampled paths
                logger.add_summary(sess.run(self.summary_op, 
                                            feed_dict={self.average_reward: 
                                                       numpy.mean([path['total_reward'] for path in paths])}))
                # Calculate discounted cumulative rewards and advantages
                samples = self.sampler.process_paths(paths, self.value_network, self.discount, self.gae_lambda,
                                                     self.sampler_center_advantage, positive_advantage=False)
                # Update policy network
                self.trpo.optimize_policy(sess, samples, logger, subsample_rate=self.subsample_rate)
                # Update value network
                self.value_network.train(paths)
                # Save the model
                if (i + 1) % 10 == 0:
                    saver.save(sess, os.path.join(self.directory, '{}.ckpt'.format(self.task)))
                # Print infos
                logger.flush()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-t', '--task', default='Swimmer', 
                        type=str, help='Tasks: Swimmer, Walker2d, Reacher, HalfCheetah, Hopper, Ant, Humanoid')
    args = parser.parse_args()
    
    device = '/{}:0'.format('cpu')
    with tf.device(device):
        trainer = Trainer(task=args.task, 
                          policy_network_hidden_sizes=(32, 32), 
                          num_epsiodes=500)
    trainer.run()
    
    