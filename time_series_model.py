#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:55:16 2022

@author: sahmaran
"""
import os
os.environ['XLA_FLAGS']= '--xla_gpu_cuda_data_dir=/usr/local/cuda-11.5'
os.listdir()
os.chdir("/home/sahmaran/Dropbox/Machines_learning")


from libraries import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout,Embedding, GRU
from tensorflow.keras.activations import relu, softmax, gelu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Concatenate, concatenate
import numpy as np

class Dense_part(Layer):
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
        inputs = tf.squeeze(inputs, axis = 2)
        l = [softmax(dense_(drop(inputs, training))) for dense_, drop in zip(self.dense, self.dropout)]
        
        return concatenate([tf.expand_dims(l, axis = 2) for l in l], axis = 2)


class Memory_part(Model):
    def __init__(self, model_dim = 256, pool_size = 1, dropout_rate = 0.5, return_states = True):
        assert pool_size >= 1, "pool size should be greater than 0"
        ###
        ###
        ###
        super().__init__();
        self.model_dim = model_dim;
        self.lstm = LSTM(model_dim, return_sequences = True, return_state = True);
        
        self.pool_size = pool_size ### the size of steps
        self.dropout = Dropout(0.5)
        self.return_state = return_states
    @tf.function
    def call(self, inputs, training = False):
        seqs, h, c = self.lstm(inputs)
        if self.return_state:
            return self.dropout(seqs[:,::-self.pool_size,:], training), h, c
        return self.dropout(seqs[:,::-self.pool_size,:], training), h

class Model_(Model):
    def __init__(self, model_dim = 256, pool_size = 5, dropout_rate = 0.5, return_states = True, dense_heads = 1):
        super().__init__()
        self.pool_size = pool_size
        self.dense_heads = dense_heads
        self.dropout_rate = dropout_rate
        self.Memory_part = Memory_part(model_dim = model_dim, pool_size = pool_size, dropout_rate = dropout_rate, return_states = return_states)
        self.compiled_ = False
        self.final_dense, self.drop_f = Sequential([Dense(20, activation = "selu"), Dense(1)]), Dropout(0.2)
    def compile_(self, lags_to_be_used):
        assert abs(lags_to_be_used/self.pool_size-  lags_to_be_used//self.pool_size) < 1e-2, "Make sure that lags_to_be_used is divisible by pool size"
        try:
            self.lags = lags_to_be_used
            
            self.dense_output = self.lags//self.pool_size
            
            self.dense = Dense_part(output = self.dense_output , dense_heads = self.dense_heads, dropout_rate = self.dropout_rate)
                
            self.compiled_ = True
            
        except Exception as a:
            print(f"Model is not compiled due to: {a}")                           
        
        
    def call(self, inputs, training = None):
        
        if not self.compiled_:
            raise CompilationError()
        assert (inputs.shape)[1] == self.lags, "Use the same lags as you promise and compiled the model"            
        d = self.dense(inputs, training)
               
        seqs, h = self.Memory_part(inputs, training)   #### sequences and the last hidden state is returned,
        res = tf.matmul(d_ ,seqs, transpose_a = True)
        res = tf.squeeze(res)
        res_fin = self.final_dense(res)
        res_fin_dropped = self.drop_f(res_fin, training)
        return d, seqs, h, res, res_fin_dropped






mod = Model_(model_dim = 20, dense_heads = 1, pool_size = 4, return_states = False)    
mod.compile_(24)
d_, k, h,res, res_fin_dropped = mod(np.random.randn(5, 24,1))

