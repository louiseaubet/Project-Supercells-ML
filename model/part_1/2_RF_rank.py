#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:49:30 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/part_1')

#####

import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import fbeta_score, matthews_corrcoef, brier_score_loss
from helpers_model import data_preparation, create_LR, create_RF

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#########

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

#########

### Data preparation
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preparation(storm, 0.2, 0.2)
ratio_meso = y_train.sum()/(len(y_train))

(y_test==0).sum()

### Normalisation for the Logistic Regression
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_valid_norm = scaler.transform(X_valid)
X_test_norm = scaler.transform(X_test)

params = {'n_estimators' : 200, # 50
          'criterion' : 'entropy',
          'max_features' : 30, # 10
          'max_depth' : 20}
rf = create_RF(**params)
rf.fit(X_train.values, y_train) 
y_pred_proba_rf = rf.predict_proba(X_test.values)[:,1]
rf

lr = create_LR(700)
lr.fit(X_train_norm, y_train)
y_pred_proba_lr = lr.predict_proba(X_test_norm)[:,1]

#########

from helpers_graph_rank import reliability_diagram, performance_diagram, \
    plot_confusion_matrix, ROC_curve
    
# Distribution of probababilities
fig = plt.figure()
plt.grid(b=True, which='major', linestyle='--')
plt.minorticks_on()
plt.hist(y_pred_proba_rf, bins=20)
plt.xlabel('Predicted probability of being mesocyclonic [-]')
plt.ylabel('Number of samples [#]')
plt.show()
fig.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/distrib_proba_RF.png', \
            dpi=150, bbox_inches='tight')
    
    
# ROC curve
fig_roc = ROC_curve(y_test, y_pred_proba_lr, y_pred_proba_rf)
fig_roc.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/ROC_rk25_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')
    
# Performance diagram 
fig_perf, thr_lr, thr_rf = performance_diagram(X_train, X_train_norm, y_train, \
            X_valid, X_valid_norm, y_valid, X_test, X_test_norm, y_test, 10, **params)
fig_perf.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/perf_diag_rk25_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')
    
fig_perf, thr_lr, thr_rf = performance_diagram(X_train, X_train_norm, y_train, \
            X_valid, X_valid_norm, y_valid, X_valid, X_valid_norm, y_valid, 10, **params)
fig_perf.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/perf_diag_valid_rk25_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')

# Reliability diagram
fig_rel = reliability_diagram(X_train, X_train_norm, y_train, \
            X_valid, X_valid_norm, y_valid, X_test, X_test_norm, y_test, 10)
fig_rel.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/rel_diag_rk25_RF_vs_LR.png', \
            dpi=150, bbox_inches='tight')

# Confusion matrix
#thr_lr = 0.303
y_test_pred_lr = np.where(y_pred_proba_lr > thr_lr, 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred_lr, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/CM_rk25_LR.png', \
            dpi=150, bbox_inches='tight')

#thr_rf = 0.353
y_test_pred_rf = np.where(y_pred_proba_rf > thr_rf, 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred_rf, normalize='all')
fig.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/CM_rk25_RF.png', \
            dpi=150, bbox_inches='tight')
    
    
### Scores
nb_models = 2
y_test_pred_lr = np.where(y_pred_proba_lr > thr_lr, 1, 0)
y_test_pred_rf = np.where(y_pred_proba_rf > thr_rf, 1, 0)
pred_array = [y_test_pred_lr, y_test_pred_rf]
precision = np.zeros((nb_models,))
recall = np.zeros((nb_models,))
f2 = np.zeros((nb_models,))
csi = np.zeros((nb_models,))
auc = np.zeros((nb_models,))
mcc = np.zeros((nb_models,))
bs = np.zeros((nb_models,))

for r in range(nb_models):
    recall[r] = recall_score(y_test, pred_array[r])
    precision[r] = precision_score(y_test, pred_array[r])
    f2[r] = fbeta_score(y_test, pred_array[r], beta=2)            
    auc[r] = roc_auc_score(y_test, pred_array[r])
    mcc[r] = matthews_corrcoef(y_test, pred_array[r])
    bs[r] = brier_score_loss(y_test, pred_array[r])
    csi[r] = 1/(1/(precision[r]) + 1/recall[r] -1)
    

# Analysis of the distribution of the TP and TN
y_pred = np.where(y_pred_proba_rf > thr_rf, 1, 0)
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
fig.savefig(code_home_path+'/outputs/part_1/1_RF_RANK/RF_rk25_base_prediction_distribution_RANK.png', \
            dpi=150, layout='tight')

