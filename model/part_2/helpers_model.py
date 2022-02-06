#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 08:37:20 2022

@author: aubet
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential, Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D

from sklearn.metrics import recall_score, precision_score

#####

### Meteorological baseline
def meteo_baseline(X_train, y_train, X_test, y_test, index_rank):
    max_rank_train = X_train[:,:,index_rank].max(axis=1)
    threshold_arr = np.linspace(min(max_rank_train), max(max_rank_train), 10)
    csi_arr = np.zeros((len(threshold_arr),))
    for i in range(len(threshold_arr)):
        y_train_pred = np.where(max_rank_train > threshold_arr[i], 1, 0)
        recall = recall_score(y_train, y_train_pred)
        precision = precision_score(y_train, y_train_pred)
        csi_arr[i] = 1/(1/(precision) + 1/recall -1)
    threshold = threshold_arr[np.nanargmax(csi_arr)]
    max_rank_test = X_test[:,:,index_rank].max(axis=1)
    y_test_pred = np.where(max_rank_test > threshold, 1, 0)
    return threshold_arr, threshold, max_rank_train, max_rank_test


def create_ML_baseline(nb_features=108, neurons=100):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(129, nb_features), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_CNN(nb_features=108, dropout_rate=0.15, nb_filter=32, neurons=200, \
               kernel_size=5):
    model = Sequential()
    model.add(Conv1D(filters=nb_filter, kernel_size=kernel_size, activation='relu', 
                 input_shape=(129, nb_features),
                 kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model




def create_CNN_old(nb_features=107, dropout_rate=0.15, nb_filter=64, neurons=100, \
               kernel_size=5):
    model = Sequential()
    model.add(Conv1D(filters=nb_filter, kernel_size=kernel_size, activation='relu', 
                 input_shape=(129, nb_features),
                 #kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



