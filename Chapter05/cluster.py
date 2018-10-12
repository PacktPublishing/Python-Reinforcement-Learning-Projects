'''
Created on 1 Jun 2017

@author: ywz
'''
PORT = 12222

def cluster_spec(num_workers, num_ps):

    cluster = {}
    port = PORT

    host = '127.0.0.1'
    cluster['ps'] = ['{}:{}'.format(host, port+i) for i in range(num_ps)]
    cluster['worker'] = ['{}:{}'.format(host, port+i+num_ps) for i in range(num_workers)]
    
    return cluster
