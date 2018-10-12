'''
Created on 21 Sep 2017

@author: ywz
'''
import multiprocessing as mp
import traceback, random
import sys, numpy
import tensorflow as tf
from joblib.pool import MemmapingPool


class SharedGlobal(object):
    pass

class StatefulPool(object):
    
    def __init__(self):
        
        self.n_parallel = 1
        self.pool = None
        self.queue = None
        self.worker_queue = None
        self.G = SharedGlobal()

    def initialize(self, n_parallel):
        
        self.n_parallel = n_parallel
        
        if self.pool is not None:
            print("Warning: terminating existing pool")
            self.pool.terminate()
            self.queue.close()
            self.worker_queue.close()
            self.G = SharedGlobal()
            
        if n_parallel > 1:
            self.queue = mp.Queue()
            self.worker_queue = mp.Queue()
            self.pool = MemmapingPool(self.n_parallel, temp_folder="/tmp")

    def run_each(self, runner, args_list=None):

        if args_list is None:
            args_list = [tuple()] * self.n_parallel
        assert len(args_list) == self.n_parallel
        
        if self.n_parallel > 1:
            results = self.pool.map_async(worker_run_each, [(runner, args) for args in args_list])
            for _ in range(self.n_parallel):
                self.worker_queue.get()
            for _ in range(self.n_parallel):
                self.queue.put(None)
            return results.get()
        else:
            return [runner(self.G, *args_list[0])]

singleton_pool = StatefulPool()

def worker_run_each(all_args):
    try:
        runner, args = all_args
        # signals to the master that this task is up and running
        singleton_pool.worker_queue.put(None)
        # wait for the master to signal continuation
        singleton_pool.queue.get()
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def worker_init(G, i):
    G.worker_id = i

def set_seed(G, seed):
    seed %= 4294967294
    random.seed(seed)
    numpy.random.seed(seed)
    tf.set_random_seed(seed)
    
def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(worker_init, [(i,) for i in range(singleton_pool.n_parallel)])
    singleton_pool.run_each(set_seed, [(123456789 + i,) for i in range(singleton_pool.n_parallel)])
    
if __name__ == "__main__":
    
    thread_num = 4
    initialize(thread_num)
    
    
    
