'''
Created on 22 Sep 2017

@author: ywz
'''
import sys
import tensorflow as tf

def delete_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)
    return path

class Logger:
    
    def __init__(self, sess, directory):
        
        self.directory = directory
        self.output_file = sys.stdout
        
        self.step = 0
        self.summary_writer = tf.summary.FileWriter(delete_dir(directory), sess.graph)
        self.print_buffer = []
    
    def clear(self):
        self.step = 0
    
    def set_step(self, step):
        self.step = step
    
    def add_summary(self, summary):
        self.summary_writer.add_summary(summary, self.step)
        summary_text = tf.Summary()
        summary_text.ParseFromString(summary)
        self.print_buffer += ["{}: {:5f}".format(v.tag, v.simple_value) for v in summary_text.value]
    
    def flush(self):
        self.summary_writer.flush()
        s = ["episode: {}".format(self.step)] + self.print_buffer
        print(', '.join(s), file=self.output_file)
        self.print_buffer = []
        
    