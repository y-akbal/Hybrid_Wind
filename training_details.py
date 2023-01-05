# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:48:56 2023

@author: yildirim.akbal
"""

import tensorflow
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard, RemoteMonitor



class training_details:
    def __init__(self):
        self.lr: float = 0.001
        self.epochs: int = 50
        self.batch_size: int = 128
        self.decay_rate: float = -0.1
        self.internal_dims: list[int] = [128, 256, 256, 256]
        self.lags: list[int] = [512, 64, 16, 8]
        self.pool_size: list[int] = [8, 4, 2, 1, 1]
        self.metric = None
        self.optimizer = Adam()
        self.lrs = LearningRateScheduler(self.lr_scheduler)
        self.csv_logger = CSVLogger("log.csv")
    @classmethod
    def from_args(cls, args):
        for key, value in vars(args):
            if value:
                pass
            else:
                self.key = value
        return cls()
    def lr_scheduler(self,epoch, lr):  #### This guy is responsible for scheduling the learning rate
        if epoch < 3:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
