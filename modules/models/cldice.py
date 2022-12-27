"""
This file is a slightly modified version of https://github.com/jocpae/clDice .
It implements fully differentiable topology-preserving morphological operations.
Title : clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling3D

def soft_erode(img):
    """
    [This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    p1 = -MaxPooling3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p2 = -MaxPooling3D(pool_size=(3, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p3 = -MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    return tf.math.minimum(tf.math.minimum(p1, p2), p3)

def soft_dilate(img):
    """
    [This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)

def soft_open(img):
    """
    [This function performs soft-open operation on a float32 image]
    Args:
        img ([float32]): [image to be soft opened]
    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode(img)
    img = soft_dilate(img)
    return img

def soft_skel(img, iters):
    """
    [summary]
    Args:
        img ([float32]): [description]
        iters ([int]): [description]
    Returns:
        [float32]: [description]
    """
    img1 = soft_open(img)
    skel = tf.nn.relu(img-img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel