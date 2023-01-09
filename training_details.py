# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:48:56 2023

@author: yildirim.akbal
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard, RemoteMonitor



class training_details:
    def __init__(self, args):
        for keys, values in args.items():
            vars(self)[keys] = values
        self.optimizer = Adam()
        self.loss = tf.keras.losses.mse
        self.lr_scheduler_ = LearningRateScheduler(self.lr_scheduler)
        self.csv_logger = CSVLogger("log.csv")
        self.callbacks = [self.lr_scheduler_, self.csv_logger]
    def lr_scheduler(self,epoch, lr):  #### This guy is responsible for scheduling the learning rate
        if epoch < 3:
            return lr
        else:
            return lr * tf.math.exp(self.decay_rate)



