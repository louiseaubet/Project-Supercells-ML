#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:50:18 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/part_1')

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score

from helpers_model import data_preparation, create_LR, create_RF
from helpers_graph_imbalance import plot_confusion_matrix, performance_diagram_imbalance

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#####

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_cleaned_extracted.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

### Data preparation
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preparation(storm, 0.2, 0.2)
ratio_meso = y_train.sum()/(len(y_train))

# Undersampling
alpha_us = 0.1
rus = RandomUnderSampler(sampling_strategy=alpha_us, random_state=42)
X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
# Oversampling
alpha_os = 0.1
ros = RandomOverSampler(sampling_strategy=alpha_os, random_state=42)
X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
# Combining oversampling and undersampling
alpha_1 = 0.05
ros = RandomOverSampler(sampling_strategy=alpha_1, random_state=42)
X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
alpha_2 = 0.1
rus = RandomUnderSampler(sampling_strategy=alpha_2, random_state=42)
X_train_ous, y_train_ous = rus.fit_resample(X_train_os, y_train_os)

### Normalisation for the Logistic Regression
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_valid_norm = scaler.transform(X_valid)
X_test_norm = scaler.transform(X_test)

### Training the different models
# Logistic Regression
lr = create_LR(700, penalty='l2')
history = lr.fit(X_train_norm, y_train)

params = {'n_estimators' : 50,
          'criterion' : 'entropy',
          'max_features' : 10,
          'max_depth' : 20,
          'random_state' : 42}
# Baseline
rf = RandomForestClassifier(**params)
rf.fit(X_train.values, y_train)
# Class weighting
rf_w = RandomForestClassifier(**params, class_weight='balanced_subsample')
rf_w.fit(X_train.values, y_train)
# Random oversampling
rf_os = RandomForestClassifier(**params)
rf_os.fit(X_train_os.values, y_train_os)
# Random undersampling
rf_us = RandomForestClassifier(**params)
rf_us.fit(X_train_us.values, y_train_us)
# Random oversampling and undersampling
rf_ous = RandomForestClassifier(**params)
rf_ous.fit(X_train_ous.values, y_train_ous)

### Predictions

nb_models = 6
models = [lr, rf, rf_w, rf_us, rf_os, rf_ous]
model_names = ['Baseline LR', 'Baseline RF', 'Class-weights',
               'Undersampling : alpha='+str(alpha_us),
               'Oversampling : alpha='+str(alpha_os), 
               'Oversampling : alpha='+str(alpha_1)+', under : '+str(alpha_2)]


# Performance diagram
fig, y_pred_proba, thr = performance_diagram_imbalance(models, model_names, X_valid,\
       X_valid_norm, y_valid, X_test, X_test_norm, y_test, nb_models)
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/perf_diag_imb_RFs.png', \
            dpi=150, bbox_inches='tight')
    
# Confusion matrices
y_test_pred = np.where(y_pred_proba[:,0] > thr[0], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_baseline_LR.png', \
            dpi=150, bbox_inches='tight')
    
y_test_pred = np.where(y_pred_proba[:,1] > thr[1], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_baseline_RF.png', \
            dpi=150, bbox_inches='tight')

y_test_pred = np.where(y_pred_proba[:,2] > thr[2], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_weights.png', \
            dpi=150, bbox_inches='tight')
    
y_test_pred = np.where(y_pred_proba[:,3] > thr[3], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_us.png', \
            dpi=150, bbox_inches='tight')

y_test_pred = np.where(y_pred_proba[:,4] > thr[4], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_os.png', \
            dpi=150, bbox_inches='tight')
    
y_test_pred = np.where(y_pred_proba[:,5] > thr[5], 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/CM_imb_RF_ous.png', \
            dpi=150, bbox_inches='tight')

    
    
# Analysis of the distribution of the TP and TN
y_test_pred_proba = models[1].predict_proba(X_test.values)[:,1]
y_pred = np.where(y_test_pred_proba > thr[1], 1, 0)
df = X_test[['RANKr']]
df['TP'] = (y_pred * y_test)
df['FP'] = ((y_test == 0) * y_pred)
df['TN'] = ((y_test==0) * (y_pred==0))
df['FN'] = (y_test * (y_pred==0))
df['tot'] = df['TP'] + df['FP'] + df['TN'] + df['FN']
df_sum = df.groupby(["RANKr"]).sum().reset_index()

fig = plt.figure()
plt.plot(df_sum['RANKr'], df_sum['TP']/df_sum['tot'], color='tab:blue', label='TP')
plt.plot(df_sum['RANKr'], df_sum['TN']/df_sum['tot'], color='tab:green', label='TN')
plt.plot(df_sum['RANKr'], df_sum['FP']/df_sum['tot'], color='tab:orange', label='FP')
plt.plot(df_sum['RANKr'], df_sum['FN']/df_sum['tot'], color='tab:red', label='FN')
plt.xlabel('RANK [-]')
plt.ylabel('Ratio of predictions [-]')
plt.legend()
plt.grid(b=True, which='major', linestyle='--')
plt.minorticks_on()
plt.show()

fig.savefig(code_home_path+'/outputs/part_1/0_RF_imbalance/RF_base_prediction_distribution_RANK.png', \
            dpi=150, layout='tight')
 