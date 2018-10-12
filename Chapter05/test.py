'''
Created on 31 May 2017

@author: ywz
'''
import time
import argparse, os, sys, signal
import tensorflow as tf
from a3c import A3C
from cluster import cluster_spec
from environment import new_environment

def shutdown(signal, frame):
    print('Received signal {}: exiting'.format(signal))
    sys.exit(128 + signal)

def test(args, server):
    
    log_dir = os.path.join(args.log_dir, '{}/train'.format(args.env))
    game, parameter = new_environment(name=args.env, test=True)
    a3c = A3C(game, log_dir, parameter.get(), agent_index=args.task, callback=game.draw)
    
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    with tf.Session(target=server.target, config=config) as sess:
        saver = tf.train.Saver()
        a3c.load(sess, saver, model_name='best_a3c_model.ckpt')
        a3c.evaluate(sess, n_episode=10, saver=None, verbose=True)

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
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        test(args, server)
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
    