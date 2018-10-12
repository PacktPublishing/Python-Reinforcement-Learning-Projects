'''
Created on 29 May 2017

@author: ywz
'''
import numpy, time, random
import argparse, os, sys, signal
import tensorflow as tf
from a3c import A3C
from cluster import cluster_spec
from environment import new_environment

def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)

def delete_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)
    return path

def shutdown(signal, frame):
    print('Received signal {}: exiting'.format(signal))
    sys.exit(128 + signal)

def train(args, server):
    
    os.environ['OMP_NUM_THREADS'] = '1'
    set_random_seed(args.task * 17)
    log_dir = os.path.join(args.log_dir, '{}/train'.format(args.env))
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    game, parameter = new_environment(args.env)
    a3c = A3C(game, log_dir, parameter.get(), agent_index=args.task, callback=None)

    global_vars = [v for v in tf.global_variables() if not v.name.startswith("local")]    
    ready_op = tf.report_uninitialized_variables(global_vars)
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    with tf.Session(target=server.target, config=config) as sess:
        saver = tf.train.Saver()
        path = os.path.join(log_dir, 'log_%d' % args.task)
        writer = tf.summary.FileWriter(delete_dir(path), sess.graph_def)
        a3c.set_summary_writer(writer)
        
        if args.task == 0:
            sess.run(tf.global_variables_initializer())
        else:
            while len(sess.run(ready_op)) > 0:
                print("Waiting for task 0 initializing the global variables.")
                time.sleep(1)
        a3c.run(sess, saver)

def main():
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-t', '--task', default=0, type=int, help='Task index')
    parser.add_argument('-j', '--job_name', default="worker", type=str, help='worker or ps')
    parser.add_argument('-w', '--num_workers', default=1, type=int, help='Number of workers')
    parser.add_argument('-l', '--log_dir', default="save", type=str, help='Log directory path')
    parser.add_argument('-e', '--env', default="demo", type=str, help='Environment')
    
    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    if args.job_name == "worker":
        server = tf.train.Server(cluster, 
                                 job_name="worker", 
                                 task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=0, 
                                                       inter_op_parallelism_threads=0)) # Use default op_parallelism_threads
        train(args, server)
    else:
        server = tf.train.Server(cluster, 
                                 job_name="ps", 
                                 task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        # server.join()
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    main()
    
    