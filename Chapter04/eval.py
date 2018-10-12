'''
Created on Apr 15, 2018

@author: ywz
'''
import os
import argparse
import tensorflow as tf
from config import DEMO
from task import Task
from dpg import DPG


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
    
    device = '/{}:0'.format('cpu')
    with tf.device(device):
        model = DPG(DEMO, task, model_dir, callback=task.render)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        model.load(sess, saver)
        model.evaluate(sess)
        

if __name__ == "__main__":
    main()
