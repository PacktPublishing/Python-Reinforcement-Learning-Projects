'''
Created on 19 Sep 2017

@author: ywz
'''
import numpy, time
import tensorflow as tf
import multiprocessing as mp
from itertools import chain
from utils import discount_cumsum
from simulator import Simulator
from utils import get_param_assign_ops, set_param_values, get_param_values
from parallel import singleton_pool, initialize


class Sampler:
    
    def __init__(self, simulator, policy):
        
        self.simulator = simulator
        self.policy = policy
        
    def rollout(self, sess, max_path_length, render=True):
        
        observations = []
        actions = []
        rewards = []
        infos = []
        path_length = 0
        observation = self.simulator.reset()
        
        while path_length < max_path_length:
            action, info = self.policy.get_action(sess, observation)
            new_observation, reward, termination = self.simulator.play(action)
            
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
        
            observation = new_observation
            path_length += 1
            if termination:
                break
            if render:
                self.simulator.render()
                time.sleep(1 / 120.0)
        
        merged_infos = {k: [] for k in infos[0].keys()}
        for info in infos:
            for k, v in info.items():
                merged_infos[k].append(v)
        for k, v in merged_infos.items():
            merged_infos[k] = numpy.array(v)
        
        samples = {'observations': numpy.array(observations),
                   'actions': numpy.array(actions),
                   'rewards': numpy.array(rewards),
                   'infos': merged_infos,
                   'total_reward': self.simulator.get_total_reward()}
        return samples

    # Calculate advantages (generalized advantage estimation)
    def process_one_path(self, path, value_network, discount, lam):
        
        values = value_network.predict(path)
        rewards = path['rewards']
        # Get discounted cumulative rewards
        path['returns'] = discount_cumsum(rewards, discount)
        # Get advantages
        vs = numpy.append(values, 0)
        deltas = rewards + discount * vs[1:] - values
        path['advantages'] = discount_cumsum(deltas, discount * lam)
        
    def process_paths(self, paths, value_network, discount, lam=1.0,
                      center_advantage=False, positive_advantage=False):

        for path in paths:
            self.process_one_path(path, value_network, discount, lam)
        
        observations = numpy.concatenate([path['observations'] for path in paths], axis=0)
        advantages = numpy.concatenate([path['advantages'] for path in paths], axis=0)
        actions = numpy.concatenate([path['actions'] for path in paths], axis=0)
        rewards = numpy.concatenate([path['rewards'] for path in paths], axis=0)
        returns = numpy.concatenate([path['returns'] for path in paths], axis=0)
        
        infos = {k: [] for k in paths[0]['infos'].keys()}
        for path in paths:
            for k, v in path['infos'].items():
                infos[k].append(v)
        for k, v in infos.items():
            infos[k] = numpy.concatenate(v, axis=0)
        
        if center_advantage:
            advantages = (advantages - numpy.mean(advantages)) / (advantages.std() + 1e-8)
        if positive_advantage:
            advantages = (advantages - numpy.min(advantages)) + 1e-8
        
        samples = {'observations': observations, 'advantages': advantages, 'actions': actions, 
                   'rewards': rewards, 'returns': returns, 'infos': infos}
        return samples

####################################################################################################

def _start_worker(G, policy_class, policy_locals, task): 
    
    G.scope = "worker_{}".format(G.worker_id)
    with tf.variable_scope(G.scope):
        policy = policy_class.copy(policy_locals)
        simulator = Simulator(task)

    G.ops = get_param_assign_ops(policy.get_params())
    G.sampler = Sampler(simulator, policy)
    G.sess = tf.Session()
    
    G.sess.__enter__()
    G.sess.run(tf.global_variables_initializer())

def _set_policy_params(G, policy_params):
    set_param_values(G.sess, G.ops[0], G.ops[1], policy_params, flatten=False)
    
def _get_policy_params(G):
    return get_param_values(G.sess, G.sampler.policy.get_params(), flatten=False)

def _generate_one_path(G, max_path_length, render):
    return G.sampler.rollout(G.sess, max_path_length, render)

def _generate_paths(G, counter, lock, max_num_samples, max_path_length, render):
    
    paths = []
    while True:
        with lock:
            if counter.value >= max_num_samples:
                return paths
            
        path = _generate_one_path(G, max_path_length, render)
        length = len(path['rewards'])
        paths.append(path)
        
        with lock:
            counter.value += length
            if counter.value >= max_num_samples:
                return paths

####################################################################################################

class ParallelSampler:
    
    def __init__(self, sampler, thread_num=2, max_path_length=numpy.inf, render=False):
        
        self.sampler = sampler
        self.thread_num = thread_num
        self.max_path_length = max_path_length
        self.render = render
        
        self.policy_class = type(self.sampler.policy)
        self.policy_locals = self.sampler.policy.get_locals()
        self.task = self.sampler.simulator.task
        
        # Initialize the multiprocessing pool
        initialize(n_parallel=thread_num)
        self.start_worker()
    
    def start_worker(self):
        singleton_pool.run_each(_start_worker, 
                                [(self.policy_class, self.policy_locals, self.task) 
                                 for _ in range(self.thread_num)])
    
    def update_policy_params(self, sess):
        policy_params = get_param_values(sess, self.sampler.policy.get_params(), flatten=False)
        singleton_pool.run_each(_set_policy_params, [(policy_params,) for _ in range(self.thread_num)])
    
    def generate_paths(self, max_num_samples):
        
        manager = mp.Manager()
        counter = manager.Value('i', 0)
        lock = manager.RLock()
            
        results = singleton_pool.run_each(_generate_paths, 
                                          [(counter, lock, max_num_samples, self.max_path_length, False) 
                                           for _ in range(self.thread_num)])

        paths = list(chain.from_iterable(r for r in results))
        return paths
    
    def truncate_paths(self, paths, max_num_samples):
        
        paths = list(paths)
        total_num_samples = sum(len(path["rewards"]) for path in paths)
        while len(paths) > 0 and total_num_samples - len(paths[-1]["rewards"]) >= max_num_samples:
            total_num_samples -= len(paths.pop(-1)["rewards"])
            
        if len(paths) > 0:
            last_path = paths.pop(-1)
            truncated_last_path = {}
            truncated_len = len(last_path["rewards"]) - (total_num_samples - max_num_samples)
            
            for k, v in last_path.items():
                if k in ["observations", "actions", "rewards"]:
                    truncated_last_path[k] = v[0:truncated_len]
                elif k in ["infos"]:
                    d = {key: item[0:truncated_len] for key, item in v.items()}
                    truncated_last_path[k] = d
                else:
                    truncated_last_path[k] = v
            
            paths.append(truncated_last_path)

        return paths
    
        
if __name__ == "__main__":
    
    from policy.gaussian_mlp import GaussianMLPPolicy
    from value.linear_fitting import LinearFitting
    
    agent = Simulator(task='Swimmer')
    input_shape = (None, agent.obsevation_dim)
    output_size = agent.action_dim
    
    policy = GaussianMLPPolicy(input_shape=input_shape,
                               output_size=output_size,
                               learn_std=True,
                               adaptive_std=False)
    value_network = LinearFitting()
    
    sampler = Sampler(agent, policy)
    parallel_sampler = ParallelSampler(sampler)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        def print_path(f, path):
            print('observations:', file=f)
            print(path['observations'], file=f)
            # print('actions:', file=f)
            # print(path['actions'], file=f)
            print('rewards:', file=f)
            print(path['rewards'], file=f)
            
            try:
                print('returns:', file=f)
                print(samples['returns'], file=f)
                print('advantages:', file=f)
                print(samples['advantages'], file=f)
            except:
                pass
            
            # print('means', file=f)
            # print(path['infos']['mean'], file=f)
            # print('log_vars', file=f)
            # print(path['infos']['log_var'], file=f)
        
        with open('log/tmp.txt', 'w') as f:
            path = sampler.rollout(sess, max_path_length=5)
            print_path(f, path)
            
            samples = sampler.process_paths([path], value_network, discount=0.9, lam=1.0,
                                            center_advantage=True)
            value_network.train(paths=[path])
            
            print("-------------------------------", file=f)
            print_path(f, samples)
            
            samples = sampler.process_paths([path], value_network, discount=0.9, lam=1.0,
                                            center_advantage=True)
            print("-------------------------------", file=f)
            print_path(f, samples)
        
        '''
        parallel_sampler.update_policy_params(sess)
        paths = parallel_sampler.generate_paths(max_num_samples=400)
        print(sum(len(path["rewards"]) for path in paths))
        paths = parallel_sampler.truncate_paths(paths, max_num_samples=400)
        print(sum(len(path["rewards"]) for path in paths))
        
        for k, v in paths[0].items():
            if k != 'infos':
                print("{}: {}".format(k, v.shape))
            else:
                for a, b in v.items():
                    print("{}: {}, {}".format(k, a, b.shape))
        for path in paths:
            print(path['total_reward'])
        '''
            