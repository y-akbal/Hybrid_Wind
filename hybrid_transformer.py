""""
Welcome home homie!
"""
# Import the library
import argparse# Create the parser
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--lr', type=float, required=True)# Parse the argument
parser.add_argument('--batch_size', type=float, required=True)# Parse the argument
parser.add_argument('--epochs', type=int, required=True)# Parse the argument
parser.add_argument('--decay_rate', type=float, required=False)# Parse the argument

args = parser.parse_args()# Print "Hello" + the user input argument

import time
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.layers import Dense, LSTM, Dropout,Embedding, GRU, BatchNormalization, Input, Attention
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.activations import relu, softmax, gelu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard, RemoteMonitor

### helper functions
"""
the last two imported
"""
try:
    from helper_functions import write_log ### this dude creates a log.txt file including some details about training
    from helper_functions import return_files, time_series_slices 
    from layers import upsample_conv, self_attention, self_attention_heads, upsampling_block, upsampling_block_with_embedding
    from training_details import training_details
    write_log("Hand crafted modules are succefully loaded dude!")
except Exception as exception:
    raise ImportError("Something is wrong with libraries see the related .py files")
    write_log("Something is wrong with libraries see the related .py files")


"""
Time for the mail model
"""


class main_model(Model):
    def __init__(self, pool_sizes = [8,4,2,1,1], lags = 512, internal_dims = [128,256, 256, 256, 256], recurent_head_dim = 128):
        super().__init__()
        assert len(internal_dims) == len(pool_sizes), "Set the internal dims and pool sizes properly"
        self.pool_sizes = pool_sizes
        self.initial_lag = lags
        self.internal_dims = internal_dims
        """  Some stuff related with lags of the time series """
        self._lags = []
        self.set_lags()        
        
        """ Recurrent head """
        self.lstm = LSTM(recurent_head_dim)
        self.dense = Dense(1)
        self.recurrent_head = Sequential([self.lstm, self.dense])
        """ The end of recurent head"""
        ####
        """The main blocks of the"""
        self.first_layer = upsampling_block_with_embedding(
            pool_size = pool_sizes[0],
            lags = lags,
            internal_dim = internal_dims[0],
            use_embed = True,
            )
        self.rest = Sequential([upsampling_block(i,int(j),k) for i,j,k in zip(self.pool_sizes[1:], self._lags[1:], self.internal_dims[1:])])
        ### pool_size = 2,lags = 16, internal_dim =  256
        """end of the main blocks"""
        ####
    def set_lags(self):
        self._lags.append(self.initial_lag)

        for i in self.pool_sizes:
            lag = self._lags[-1]/i
            if lag.is_integer():
                self._lags.append(lag)
            else:
                raise ValueError("Jeeezzz, somethin' went wrong bro, can't ya get it????? Check with the pool sizes")
        
    @tf.function(jit_compile = True)
    def call(self, inputs, training = None):
        x = self.first_layer(inputs, training)
        x = self.rest(x, training)
        x = self.recurrent_head(x)
        return x



""" Time for the main function """

if __name__ == "__main__":
    
    write_log("Training started") 
    td = training_details()
    
    
    
    
    
    _, data_ = return_files()
    print("Splitting the above series...")
    slicer = time_series_slices(512)
    train, test = slicer.fit_transform(data_)
    X_train, y_train,  train_class = train
    X_test, y_test,  test_class = test
    
    print("Splitting into test and train sets is done!")
    
    
    
    model = main_model()
    model([np.random.randn(2,512,1),np.array([1,2])])
    
    write_log("Training ended")
    #return_files()
    #
    #
    #
    #
    #
    #
