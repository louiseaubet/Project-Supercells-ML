#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:49:30 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_1')

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from helpers_model import data_preparation, create_LR, create_RF
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#########

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt'
cvx_path = data_home_path+'/processed/new/TRT_storm1_cvx_rk25_cleaned.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)
storm_cvx = pd.read_csv(cvx_path, sep=',', header=0, index_col=0)

#########

def data_preparation_cvx(data, test_size=0.2, valid_size=0.2):
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

### Data preparation
X_train_cvx, X_valid_cvx, X_test_cvx, y_train_cvx, y_valid_cvx, y_test_cvx = \
    data_preparation_cvx(storm_cvx, 0.2, 0.2)
        
X_train = X_train_cvx.drop('convexity', axis=1)
X_valid = X_valid_cvx.drop('convexity', axis=1)
X_test = X_test_cvx.drop('convexity', axis=1)

### Normalisation for the Logistic Regression
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_cvx)
X_valid_norm = scaler.transform(X_valid_cvx)
X_test_norm = scaler.transform(X_test_cvx)

params = {'n_estimators' : 200, # 50
          'criterion' : 'entropy',
          'max_features' : 30, # 10
          'max_depth' : 20}

lr_cvx = create_LR()
lr_cvx.fit(X_train_norm, y_train_cvx)
y_pred_proba_lr_cvx = lr_cvx.predict_proba(X_test_norm)[:,1]

rf = create_RF(**params)
rf.fit(X_train.values, y_train_cvx) 
y_pred_proba_rf = rf.predict_proba(X_test.values)[:,1]

rf_cvx = create_RF(**params)
rf_cvx.fit(X_train_cvx.values, y_train_cvx) 
y_pred_proba_rf_cvx = rf_cvx.predict_proba(X_test_cvx.values)[:,1]


#########

from helpers_graph_cvx import performance_diagram, \
    plot_confusion_matrix, ROC_curve

# ROC curve
fig_roc = ROC_curve(y_test_cvx, y_pred_proba_lr_cvx, y_pred_proba_rf, y_pred_proba_rf_cvx)
fig_roc.savefig(code_home_path+'/outputs/part_1/2_RF_convexity/ROC_cvx_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')
    
# Performance diagram 
fig_perf, thr_lr_cvx, thr_rf, thr_rf_cvx = performance_diagram(
                        X_train, X_train_norm, X_train_cvx, y_train_cvx,
                        X_valid, X_valid_norm, X_valid_cvx, y_valid_cvx,
                        X_test, X_test_norm, X_test_cvx, y_test_cvx, 10, **params)
fig_perf.savefig(code_home_path+'/outputs/part_1/2_RF_convexity/perf_diag_cvx_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')

    
# Confusion matrix
y_test_pred_lr_cvx = np.where(y_pred_proba_lr_cvx > thr_lr_cvx, 1, 0)
fig = plot_confusion_matrix(y_test_cvx, y_test_pred_lr_cvx, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/2_RF_convexity/CM_cvx_LRcvx.png', \
            dpi=150, bbox_inches='tight')
    
y_test_pred_rf = np.where(y_pred_proba_rf > thr_rf, 1, 0)
fig = plot_confusion_matrix(y_test_cvx, y_test_pred_rf, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/2_RF_convexity/CM_cvx_RF.png', \
            dpi=150, bbox_inches='tight')

y_test_pred_rf_cvx = np.where(y_pred_proba_rf_cvx > thr_rf_cvx, 1, 0)
fig = plot_confusion_matrix(y_test_cvx, y_test_pred_rf_cvx, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/2_RF_convexity/CM_cvx_RFcvx.png', \
            dpi=150, bbox_inches='tight')
    
