#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:46:25 2022

@author: aubet
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve
from sklearn.metrics import recall_score, precision_score, auc
from matplotlib.colors import LogNorm

from helpers_model import create_LR, create_RF

#####

# Confusion matrix
def plot_confusion_matrix(y_test, y_test_pred_proba, normalize=None):
    #sns.set_palette(sns.color_palette("viridis"))
    cf_matrix = confusion_matrix(y_test, y_test_pred_proba, normalize=normalize)
    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, cbar=False, norm=LogNorm(), fmt='.6%')
    plt.xlabel('Predicted class')
    plt.ylabel('Real class')
    plt.show()
    return fig



# Performance diagram
def performance_diagram_imbalance(models, model_names, X_train, X_train_norm, y_train, X_test, \
                                  X_test_norm ,y_test, nb_models):
    threshold = np.linspace(1e-8, 0.999, 100)
    precision = np.zeros((len(threshold),nb_models))
    recall = np.zeros((len(threshold),nb_models))
    precision_train = np.zeros((len(threshold),nb_models))
    recall_train = np.zeros((len(threshold),nb_models))
    csi_train = np.zeros((len(threshold),nb_models))
    y_train_pred_proba = np.zeros((len(y_train),nb_models))
    y_test_pred_proba = np.zeros((len(y_test),nb_models))
    for r in range(nb_models-1):
        y_train_pred_proba[:,r+1] = models[r+1].predict_proba(X_train.values)[:,1]
        y_test_pred_proba[:,r+1] = models[r+1].predict_proba(X_test.values)[:,1]
    y_train_pred_proba[:,0] = models[0].predict_proba(X_train_norm)[:,1]
    y_test_pred_proba[:,0] = models[0].predict_proba(X_test_norm)[:,1]
    for r in range(nb_models):
        for i in range(len(threshold)):
            y_train_pred = np.where(y_train_pred_proba[:,r] > threshold[i], 1, 0)
            y_test_pred = np.where(y_test_pred_proba[:,r] > threshold[i], 1, 0)
            recall[i,r] = recall_score(y_test, y_test_pred)
            precision[i,r] = precision_score(y_test, y_test_pred)
            recall_train[i,r] = recall_score(y_train, y_train_pred)
            precision_train[i,r] = precision_score(y_train, y_train_pred)
            csi_train[i,r] = 1/(1/(precision_train[i,r]) + 1/(recall_train[i,r]) -1)
            
    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    xx, yy = np.meshgrid(x, y)
    csi_map = 1/(1/(xx) + 1/yy -1)
    
    #csi_max = np.nanmax(csi_train, axis=0)
    index_max = np.nanargmax(csi_train, axis=0)
    thr = threshold[index_max]

    fig, ax = plt.subplots()
    ct = ax.contourf(xx, yy, csi_map, 15, cmap='Blues')
    cbar = fig.colorbar(ct)
    cbar.set_label('CSI')
    ax.autoscale(False)
    plt.grid(b=True, which='major', linestyle='--')
    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']
    for r in range(nb_models):
        ax.plot(precision[:,r], recall[:,r], label=model_names[r], color=colors[r], \
                linestyle='-')
        ax.scatter(precision[index_max[r],r], recall[index_max[r],r], color=colors[r],\
                linestyle='-', marker='v', s=70)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.legend(fontsize=8)
    plt.minorticks_on()
    plt.axis([0,1.0,0,1.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return fig, y_test_pred_proba, thr


