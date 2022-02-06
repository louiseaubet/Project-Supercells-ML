#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:29:39 2021

@author: aubet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#####


def data_preparation(data, test_size=0.2, valid_size=0.2):
    # Separation between features and targets
    labels = np.where((data['meso'] == 1), 1, 0)    
    features = data.drop(['meso'], axis=1)

    # Train-test splitting
    features_sh, labels_sh = shuffle(features, labels, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels,
                                                    test_size=test_size, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train,
                                                    test_size=valid_size, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_RF(**params):
    rf = RandomForestClassifier(**params)
    return rf

def create_LR(n_max=500, penalty='l2'):
    lr = LogisticRegression(max_iter=n_max, penalty=penalty)
    return lr
