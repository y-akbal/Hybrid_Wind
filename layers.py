
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:53:20 2022

@author: sahmaran
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM, Dropout,Embedding, GRU, BatchNormalization, Input, Attention
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.activations import relu, softmax, gelu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, CSVLogger
import numpy as np




"""
Time for layers
"""

class upsample_conv(Layer):
    def __init__(self, pool_size = 4, lags = 256, internal_dim = 128, activation = "gelu", use_embed = False):
        assert (lags/pool_size).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.conv = Conv1D(filters = internal_dim, kernel_size = pool_size, strides = pool_size, use_bias = True)
        self.activation = tf.keras.activations.get(activation)
        self.norm = BatchNormalization()
        self.use_embed = use_embed
        if use_embed:
            self.embed = Embedding(int(lags/pool_size), internal_dim)
            self.list = tf.constant([i for i in range(int(lags/pool_size))])
    @tf.function(jit_compile= True)
    def call(self, inputs, training = None):
        x = self.conv(inputs)
        if self.use_embed:
            x += self.embed(self.list)
        x = self.activation(x)
        x = self.norm(x, training)
        return x


class self_attention(Layer):
    def __init__(self, heads = 5, causal = True, dropout = 0.3, **kwargs):
        super(self_attention, self).__init__(**kwargs)
        self.causal = causal 

        self.dropout = Dropout(dropout)
    def build(self, input_shape):
        
        input_shape = input_shape[0]
        shape = input_shape[-2], input_shape[-2]
        initializer = tf.keras.initializers.glorot_normal()#### we in particular ortogonal initialization, in the case that it is needed it will train it!
        initial_value = initializer(shape = shape)
        self.kernel = tf.Variable(initial_value = initial_value, trainable = True)
        if self.causal: ### this part is used to kill attention of future to past, 
            minf = -tf.constant(20000.0)  ### take this dude to kill softmax maybe a little bit smaller.
            mask = tf.fill(shape, minf)
            self.upper_m = minf - tf.linalg.band_part(mask, num_lower = -1, num_upper = 0)
    @tf.function(jit_compile = True)
    def call(self, inputs, training = None):
        if training:
            inputs = self.dropout(inputs, training) ### dropout is applied in the begining of the layer
        input_1 = tf.matmul(self.kernel, inputs[0])
        input_2 = tf.matmul(self.kernel, inputs[1])
        att_scores = tf.matmul(input_1, input_2, transpose_b = True)  ### we get the attention scores
        d_k = (input_1.shape[-2])**0.5 ####normalizing factor is here
        if self.causal:
            return tf.nn.softmax((att_scores+self.upper_m)/d_k) @ input_1
        
        return  tf.nn.softmax(att_scores/d_k) @ input_1 
    
    
class self_attention_heads(Layer):
    def __init__(self, heads = 5, causal = False, dropout = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.causal = causal 
        self.heads = heads
        self.dropout = Dropout(dropout)
        self.conv2d = Conv2D(1, kernel_size = 1, strides = 1, use_bias= True, kernel_initializer= tf.keras.initializers.Constant(
    value=1/self.heads))
        
    def build(self, input_shape, **kwargs):
       
        input_shape = input_shape[0]
        shape = self.heads, input_shape[-2], input_shape[-2]
        initializer = tf.keras.initializers.Orthogonal() #### we in particular ortogonal initialization, in the case that it is needed it will train it!
        initial_value = initializer(shape = shape)
        
        self.kernel = tf.Variable(initial_value = initial_value, trainable = True)
        
        if self.causal: ### this part is used to kill attention of future to past, 
            minf = -tf.constant(20000.0)  ### take this dude to kill softmax maybe a little bit smaller.
            mask = tf.fill(shape, minf)
            self.upper_m = minf - tf.linalg.band_part(mask, num_lower = -1, num_upper = 0)
            
    @tf.function(jit_compile = True)
    def call(self, inputs, training = None):
        if training:
            inputs = self.dropout(inputs, training) ### dropout is applied in the begining of the layer
        inputs_ = [0 for i in range(len(inputs))]
        inputs_[0] = tf.expand_dims(inputs[0], axis = -3)
        inputs_[1] = tf.expand_dims(inputs[1], axis = -3)
        sim1 = self.kernel @ inputs_[0]
        sim2 = self.kernel @ inputs_[1]
        att_scores = tf.matmul(sim1, sim2, transpose_b = True)
        if self.causal:
            att_scores += self.upper_m
        softmaxed = tf.nn.softmax(att_scores, axis = -2)
        similarity_heads = softmaxed @ inputs_[1]
        transposed = tf.transpose(similarity_heads, [0, 2, 3, 1])
        return tf.squeeze(self.conv2d(transposed), -1)      
    
    
    
    


class upsampling_block(Model):
    def __init__(self, pool_size, lags, internal_dim, attention_heads = 5, dropout_rate = 0.4, causal = False, **kwargs):
        super().__init__()
        self.upsample_conv = upsample_conv(pool_size = pool_size,
                                           lags = lags,
                                           internal_dim = internal_dim,
                                           **kwargs)
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(internal_dim)
        self.att = self_attention_heads(heads = attention_heads,
                                        causal = causal,
                                        dropout= dropout_rate,
                                        )
        self.batch_norm = BatchNormalization()
    @tf.function(jit_compile = True)
    def call(self, inputs, training = None):
        t = self.upsample_conv(inputs, training)
        x = self.dropout(t, training)
        x = self.dense(x)
        x = self.att([x,x], training)+t #### residual connection here!
        x = self.batch_norm(x, training)
        return x

class upsampling_block_with_embedding(Model):
    def __init__(self, pool_size, lags, 
                 internal_dim, dropout_rate = 0.4, 
                 number_embeedings = 8, ### this is the number of time series to be used should be adjusted properly
                 causal = True, **kwargs):
        super().__init__()
        self.upsample_conv = upsample_conv(pool_size = pool_size,
                                           lags = lags,
                                           internal_dim = internal_dim,
                                           use_embed = True)
        self.embedding = Embedding(number_embeedings, internal_dim)
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(internal_dim)
        self.att = Attention(causal = causal)
        self.batch_norm = BatchNormalization()
    @tf.function(jit_compile = True)
    def call(self, inputs, training = None):
        inputs_1, inputs_2 = inputs[0], inputs[1]
        embeddings = self.embedding(inputs_2)

        t = self.upsample_conv(inputs_1, training)
        x = self.dropout(t, training)
        x = self.dense(x)+tf.expand_dims(embeddings, 1)### yet another residual connection here!
        x = self.att([x,x], training) + t #### residual connection here!
        x = self.batch_norm(x, training)
        return x

"""
Create the main model
"""