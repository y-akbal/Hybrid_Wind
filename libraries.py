#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:56:38 2022

@author: sahmaran
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout,Embedding
from tensorflow.keras.activations import relu, softmax, gelu
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy




if __name__ == "main":
    print("s")


