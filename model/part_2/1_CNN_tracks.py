#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 08:22:13 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/part_2')

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import fbeta_score, matthews_corrcoef, brier_score_loss
from keras.callbacks import EarlyStopping

from helpers_preprocessing import split_with_IDs, prepare_sets, \
    random_oversampling, normalisation
from helpers_model import meteo_baseline, create_ML_baseline, create_CNN
from helpers_graphs import learning_curve, plot_confusion_matrix, F2_score_plot, \
    ROC_curve, precision_recall_curve, reliability_diagram, performance_diagram
    
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#####

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_stormtracks_rk25_pad.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)
                    
### Data preparation

### Split into train and test sets. Be careful of keeping same IDs together
train, valid, test = split_with_IDs(storm, 0.2, 0.1)

### Normalize the dataset : min-max
train_n, valid_n, test_n = normalisation(train, valid, test)

### Prepare train and test sets                
X_train, y_train, feature_list = prepare_sets(train_n)
X_valid, y_valid, _ = prepare_sets(valid_n)
X_test, y_test, _ = prepare_sets(test_n)

# Parameters
n_timesteps = 129
n_features = X_train.shape[2]
index_rank = feature_list.index("RANKr")

####################

# Create and train the models

### Meteorological baseline
thr_arr_meteo_b, thr_meteo_b, max_rank_train, max_rank_test =\
                    meteo_baseline(X_train, y_train, X_test, y_test, index_rank)
y_test_pred_meteo_base = np.where(max_rank_test > thr_meteo_b, 1, 0)

### ANN Baseline
baseline = create_ML_baseline(nb_features=n_features, neurons=100)
baseline.summary()
history_ANN_base = baseline.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
y_valid_pred_proba_base = baseline.predict(X_valid)
y_test_pred_proba_base = baseline.predict(X_test)

### CNN model
model = create_CNN(nb_features=n_features, dropout_rate=0.15, nb_filter=32,\
                   neurons=200, kernel_size=5)
history_CNN = model.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
model.summary()
y_valid_pred_proba_CNN = model.predict(X_valid)
y_test_pred_proba_CNN = model.predict(X_test)


### Performance diagram
fig, thr_b, thr_ann, thr_cnn = performance_diagram(X_train, y_train, X_valid, y_valid, 
         X_test, y_test, max_rank_train, max_rank_test, thr_arr_meteo_b, 10)
    
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/perf_diag_CNN_vs_ANN.png', \
            dpi=150, bbox_inches='tight')

    
fig_v, thr_b, thr_ann, thr_cnn = performance_diagram(X_train, y_train, X_valid, y_valid, 
         X_valid, y_valid, max_rank_train, max_rank_test, thr_arr_meteo_b, 10)
    
fig_v.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/perf_diag_valid_CNN_vs_ANN.png', \
            dpi=150, bbox_inches='tight')
    

####################

### Evaluate the model 

### Distribution of probabilities
fig = plt.figure()
plt.grid(b=True, which='major', linestyle='--')
plt.minorticks_on()
plt.hist(y_test_pred_proba_CNN, bins=15)
plt.xlabel('Predicted probability of being mesocyclonic [-]')
plt.ylabel('Number of samples [#]')
plt.show()
fig.savefig(code_home_path+'/outputs/part_2/models/distrib_proba_CNN.png', \
            dpi=150, bbox_inches='tight')
    
### ROC curve
fig = ROC_curve(y_test, y_test_pred_proba_base, y_test_pred_proba_CNN)
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/ROC_CNN_vs_ANN.png', \
            dpi=150, bbox_inches='tight')

fig = ROC_curve(y_valid, y_valid_pred_proba_base, y_valid_pred_proba_CNN)
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/ROC_valid_CNN_vs_ANN.png', \
            dpi=150, bbox_inches='tight')
    
    
### Reliability diagram
fig = reliability_diagram(X_train, y_train, X_valid, y_valid, X_test, y_test, 2)
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/rel_diag_CNN_vs_ANN.png', \
            dpi=150, bbox_inches='tight')
    

### Plot learning curves
fig = learning_curve(history_ANN_base)
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/LC_ANN_base.png', \
            dpi=150, bbox_inches='tight')

fig = learning_curve(history_CNN)
fig.savefig(code_home_path+'/outputs/part_2/0_CNN_performance/LC_CNN.png', \
            dpi=150, bbox_inches='tight')

### Scores
nb_models = 3
y_test_pred_base = np.where(y_test_pred_proba_base > thr_ann, 1, 0)
y_test_pred_CNN = np.where(y_test_pred_proba_CNN > thr_cnn, 1, 0)
pred_array = [y_test_pred_meteo_base, y_test_pred_base, \
              y_test_pred_CNN]
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
    

### Plot confusion matrix
fig = plot_confusion_matrix(y_test, y_test_pred_meteo_base, normalize='all')
fig.savefig('/home/aubet/plots/part_2/models/CM_meteo_base.png', \
            dpi=150, bbox_inches='tight')

y_test_pred_base = np.where(y_test_pred_proba_base > thr_ann, 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred_base, normalize='all')
fig.savefig('/home/aubet/plots/part_2/models/CM_ANN_base.png', \
            dpi=150, bbox_inches='tight')

y_test_pred_CNN = np.where(y_test_pred_proba_CNN > thr_cnn, 1, 0)
fig = plot_confusion_matrix(y_test, y_test_pred_CNN, normalize='all')
fig.savefig('/home/aubet/plots/part_2/models/CM_CNN.png', \
            dpi=150, bbox_inches='tight')

    
    
    
# Analysis of the distribution of the TP and TN
y_test_pred_proba = model.predict(X_test)
y_pred = np.where(y_test_pred_proba > thr_cnn, 1, 0).reshape((len(y_test)))
df = pd.DataFrame(max_rank_test, columns=['max_RANK'])
df['TP'] = (y_pred * y_test)
df['FP'] = ((y_test == 0) * y_pred)
df['TN'] = ((y_test==0) * (y_pred==0))
df['FN'] = (y_test * (y_pred==0))
df['tot'] = df['TP'] + df['FP'] + df['TN'] + df['FN']
df_sum = df.groupby(["max_RANK"]).sum().reset_index()

fig = plt.figure()
plt.plot(df_sum['max_RANK'], df_sum['TP']/df_sum['tot'], color='tab:blue', label='TP')
plt.plot(df_sum['max_RANK'], df_sum['TN']/df_sum['tot'], color='tab:green', label='TN')
plt.plot(df_sum['max_RANK'], df_sum['FP']/df_sum['tot'], color='tab:orange', label='FP')
plt.plot(df_sum['max_RANK'], df_sum['FN']/df_sum['tot'], color='tab:red', label='FN')
plt.xlabel('Maximum RANK [-]')
plt.ylabel('Ratio of predictions [-]')
plt.legend()
plt.grid(b=True, which='major', linestyle='--')
plt.minorticks_on()
plt.show()

fig.savefig("/home/aubet/plots/part_2/post-analysis/CNN_base_prediction_distribution_max_RANK.png", \
            dpi=150, layout='tight')
 
