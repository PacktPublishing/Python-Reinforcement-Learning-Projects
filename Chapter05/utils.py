'''
Created on Nov 8, 2016

@author: a0096049
'''
import math, random
import numpy, cv2
import skimage.transform
import tensorflow as tf


def preprocess_image(im, image_shape=(110, 84), crop_shape=84, crop_part='down'):
    
    im = skimage.transform.resize(im, image_shape, preserve_range=True)
    
    half = int(crop_shape / 2)
    h, w = im.shape
    if crop_part == 'center':
        im = im[h//2-half:h//2+half, w//2-half:w//2+half]
    if crop_part == 'down':
        im = im[h-crop_shape:h, w//2-half:w//2+half]

    return numpy.asarray(im, dtype=numpy.uint8)

def cv2_resize_image(image, resized_shape=(84, 84), method='crop', crop_offset=8):
        
        height, width = image.shape
        resized_height, resized_width = resized_shape
        
        if method == 'crop':
            h = int(round(float(height) * resized_width / width))
            resized = cv2.resize(image, (resized_width, h), interpolation=cv2.INTER_LINEAR)
            crop_y_cutoff = h - crop_offset - resized_height
            cropped = resized[crop_y_cutoff : crop_y_cutoff + resized_height, :]
            return numpy.asarray(cropped, dtype=numpy.uint8)
        elif method == 'scale':
            return numpy.asarray(cv2.resize(image, (resized_width, resized_height), 
                                            interpolation=cv2.INTER_LINEAR), dtype=numpy.uint8)
        else:
            raise ValueError('Unrecognized image resize method.')

def log_uniform(low, high):
    return math.exp(random.uniform(math.log(low), math.log(high)))

def update_target_graph(from_vars, to_vars):
    
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
        
    return op_holder

def create_optimizer(method, learning_rate, rho, epsilon):
    
    if method == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
                                        decay=rho,
                                        epsilon=epsilon)
    elif method == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=rho)
    else:
        raise
    
    return opt

    