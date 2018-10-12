'''
Created on Mar 25, 2018

@author: ywz
'''

ATARI = {
    'network_type': 'cnn',
    'gamma': 0.99,
    'batch_size': 32,
    'num_episode': 500000,
    'capacity': 1000000,
    'epsilon_decay': 1000000,
    'epsilon_min': 0.1,
    'num_frames': 4,
    'num_nullops': 5,
    'time_between_two_copies': 10000,
    'input_scale': 255.0,
    'update_interval': 1,
    'T': 100000,
    
    'learning_rate': 2e-4,
    'optimizer': 'rmsprop',
    'rho': 0.99,
    'rmsprop_epsilon': 1e-6,
    
    'log_dir': 'log/'
}


DEMO = {
    'network_type': 'mlp',
    'gamma': 0.7,
    'batch_size': 32,
    'num_episode': 40,
    'capacity': 20000,
    'epsilon_decay': 100000,
    'epsilon_min': 0.1,
    'num_frames': 1,
    'num_nullops': 2,
    'time_between_two_copies': 1000,
    'input_scale': 1.0,
    'update_interval': 1,
    'T': 1000000,
    
    'learning_rate': 0.5e-2,
    'optimizer': 'momentum',
    'rho': 0.9,
    'rmsprop_epsilon': 1e-6,
    
    'log_dir': 'log/'
}
