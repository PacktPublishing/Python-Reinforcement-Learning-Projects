'''
Created on Apr 11, 2018

@author: ywz
'''
DEMO = {
    'gamma': 0.99,
    'history_len': 2,
    'num_episode': 3000,
    'capacity': 100000,
    'epsilon_decay': 100000,
    'epsilon_min': 0.0,
    'time_between_two_copies': 2000,
    'update_interval': 1,
    'T': 1000000,
    
    'batch_size': 64,
    'learning_rate': 1e-4,
    'tau': 0.9,
    'optimizer': 'adam',
    'rho': 0.99,
    'log_dir': 'log/'
}
