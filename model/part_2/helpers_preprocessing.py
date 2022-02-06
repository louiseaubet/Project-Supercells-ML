#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:16:01 2021

@author: aubet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#####

def split_with_IDs(data, ratio_test, ratio_valid):
    meso_ID_list = data.loc[data['mesostorm'] == 1]['ID'].unique()
    #np.random.shuffle(meso_ID_list)
    nomeso_ID_list = data.loc[data['mesostorm'] == 0]['ID'].unique()
    #np.random.shuffle(nomeso_ID_list)
    
    # Class 1 : mesocyclones
    meso_train_ID_size = int(len(meso_ID_list) * (1-ratio_test-ratio_valid)) + 1
    meso_valid_ID_size = int(len(meso_ID_list) * ratio_valid)
    
    meso_train_ID_list = list(meso_ID_list[0:meso_train_ID_size])
    meso_valid_ID_list = list(meso_ID_list[meso_train_ID_size:(meso_train_ID_size+meso_valid_ID_size)])
    meso_test_ID_list = list(meso_ID_list[(meso_train_ID_size+meso_valid_ID_size):])

    # Class 0 : non-mesocyclones
    nomeso_train_ID_size = int(len(nomeso_ID_list) * (1-ratio_test-ratio_valid)) + 1
    nomeso_valid_ID_size = int(len(nomeso_ID_list) * ratio_valid)
    
    nomeso_train_ID_list = list(nomeso_ID_list[0:nomeso_train_ID_size])
    nomeso_valid_ID_list = list(nomeso_ID_list[nomeso_train_ID_size:(nomeso_train_ID_size+nomeso_valid_ID_size)])
    nomeso_test_ID_list = list(nomeso_ID_list[(nomeso_train_ID_size+nomeso_valid_ID_size):])
    
    # Assemble ID lists
    train_ID_list = meso_train_ID_list + nomeso_train_ID_list
    valid_ID_list = meso_valid_ID_list + nomeso_valid_ID_list
    test_ID_list = meso_test_ID_list + nomeso_test_ID_list
    
    train = data.loc[data['ID'].isin(train_ID_list)].reset_index(drop=True)
    valid = data.loc[data['ID'].isin(valid_ID_list)].reset_index(drop=True)
    test = data.loc[data['ID'].isin(test_ID_list)].reset_index(drop=True)
        
    train_sorted = train.sort_values(['ID','time'], ascending=True)
    valid_sorted = valid.sort_values(['ID','time'], ascending=True)
    test_sorted = test.sort_values(['ID','time'], ascending=True)
    
    return train_sorted, valid_sorted, test_sorted

def prepare_sets(data):
    ID_list = data['ID'].unique()
    X = []
    y = []
    for i in range(len(ID_list)):
        data_ID = data.loc[data['ID'] == ID_list[i]]
        y_col = 'mesostorm'
        X_col = list(data.columns)
        unwanted_el = ['mesostorm', 'ID', 'time']
        X_col = [el for el in X_col if el not in unwanted_el]
        X.append(data_ID.loc[:,X_col].values)
        y.append(data_ID[y_col].iloc[0])
    X = np.array(X)
    y = np.array(y)
    return X, y, X_col

def random_oversampling(X_train, y_train, ratio=0.2):
    X_train_os = X_train
    y_train_os = y_train
    ratio = np.sum(y_train_os == 1)/np.sum(y_train_os == 0)
    indices_meso = np.array(np.where(y_train_os == 1)).flatten()
    ratio_f = ratio
    while(ratio < ratio_f):
        index_to_copy = np.random.choice(indices_meso, size=1)
        sample_to_copy = X_train[index_to_copy,:,:]
        X_train_os = np.concatenate((X_train_os, sample_to_copy), axis=0)
        y_train_os = np.append(y_train_os, 1)
        ratio = np.sum(y_train_os == 1)/np.sum(y_train_os == 0)
    return X_train_os, y_train_os

def normalisation(train, valid, test):
    train_n = train.copy()
    valid_n = valid.copy()
    test_n = test.copy()
    
    cols_to_norm = [name for name in train_n.columns if not (name.startswith("cat_"))]
    cols_to_norm.remove('mesostorm')
    
    scaler = MinMaxScaler([0,1])
    scaler = scaler.fit(train[cols_to_norm])
    train_n[cols_to_norm] = scaler.transform(train_n[cols_to_norm])
    valid_n[cols_to_norm] = scaler.transform(valid_n[cols_to_norm])
    test_n[cols_to_norm] = scaler.transform(test_n[cols_to_norm])

    # Change padding from NaN to an unused value
    train_n = train_n.replace(np.NaN, 0) #-0.2
    test_n = test_n.replace(np.NaN, 0)
    valid_n = valid_n.replace(np.NaN, 0)
    return train_n, valid_n, test_n

