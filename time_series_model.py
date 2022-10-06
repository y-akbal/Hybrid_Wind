#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:55:16 2022

@author: sahmaran
"""
import os
os.environ['XLA_FLAGS']= '--xla_gpu_cuda_data_dir=/usr/local/cuda-11.5'

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout,Embedding
from tensorflow.keras.activations import relu, softmax, gelu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Concatenate, concatenate
import numpy as np

class Dense_mod(Layer):
    def __init__(self, output:int, dense_heads = 3, dropout_rate = 0.5):
        super().__init__()
        self.output_dim = output
        self.dense_heads = dense_heads
        self.dense = [Dense(self.output_dim) for i in range(self.dense_heads)]
        self.dropout_rate = dropout_rate 
        self.dropout = [Dropout(self.dropout_rate) for _ in self.dense]
        self.concate = Concatenate
    
    @tf.function(jit_compile= True)
    def call(self, inputs, training = None):
        l = [softmax(dense_(drop(inputs, training))) for dense_, drop in zip(self.dense, self.dropout)]
        if self.dense_heads == 1:
            return tf.squeeze(l)
        return concatenate([tf.expand_dims(l, axis = 2) for l in l], axis = 2)

l = Dense_mod(10,dense_heads = 2)
l(np.random.randn(1, 10))