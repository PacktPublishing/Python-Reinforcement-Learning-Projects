'''
Created on Mar 28, 2018

@author: ywz
'''
import os
import argparse
import tensorflow as tf
from q_learning import DQN
from config import ATARI, DEMO
from environment import new_atari_game, new_demo


def main():
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-g', '--game', default='demo', type=str, help='Game')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='Device')
    args = parser.parse_args()
    
    rom = args.game
    if rom == 'demo':
        game = new_demo()
        conf = DEMO
    else:
        game = new_atari_game(rom)
        conf = ATARI

    model_dir = os.path.join(conf['log_dir'], rom)
    device = '/{}:0'.format(args.device)
    with tf.device(device):
        dqn = DQN(conf, game, model_dir, callback=game.draw)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        dqn.load(sess, saver)
        dqn.evaluate(sess)
        

if __name__ == "__main__":
    main()
