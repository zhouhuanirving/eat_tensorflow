# coding=utf-8


'''
Author:zhouhuan
Email:18832832911@139.com

data:2020/12/11/011 19:31
desc:
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

import os
os.environ["CUDA_VISIBLE_DEVICES"]= " "
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images































