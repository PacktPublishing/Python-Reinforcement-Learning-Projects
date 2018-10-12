'''
Created on Apr 12, 2018

@author: ywz
'''
import os
import argparse
import tensorflow as tf
from config import DEMO
from task import Task
from dpg import DPG


def delete_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)
    return path


def main():
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-t', '--task', default='CartPole-v0', 
                        type=str, help='Tasks: CartPole-v0, Pendulum-v0, Acrobot-v1')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='Device: cpu, gpu')
    args = parser.parse_args()
    
    task = Task(args.task)
    log_dir = os.path.join(DEMO['log_dir'], '{}/train'.format(args.task))
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    model_dir = os.path.join(DEMO['log_dir'], args.task)
    
    device = '/{}:0'.format(args.device)
    with tf.device(device):
        model = DPG(DEMO, task, model_dir, callback=None)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(delete_dir(log_dir), sess.graph_def)
        model.set_summary_writer(summary_writer=writer)
        
        sess.run(tf.global_variables_initializer())
        model.train(sess, saver)
        

if __name__ == "__main__":
    main()

