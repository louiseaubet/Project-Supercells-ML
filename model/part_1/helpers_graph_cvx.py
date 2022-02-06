#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:26:10 2021

@author: aubet

Some helper functions

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
def plot_confusion_matrix(y_test, y_test_pred, normalize=None):
    cf_matrix = confusion_matrix(y_test, y_test_pred, normalize=normalize)
    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, cbar=False, norm=LogNorm())
    plt.xlabel('Predicted class')
    plt.ylabel('Real class')
    plt.show()
    return fig

# ROC curve
def ROC_curve(y_test, y_pred_proba_lr, y_pred_proba_rf, y_pred_proba_rf_cvx):
    fpr_lr, recall_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    fpr_rf, recall_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    fpr_rf_cvx, recall_rf_cvx, _ = roc_curve(y_test, y_pred_proba_rf_cvx)
    roc_auc_lr = auc(fpr_lr, recall_lr)
    roc_auc_rf = auc(fpr_rf, recall_rf)
    roc_auc_rf_cvx = auc(fpr_rf_cvx, recall_rf_cvx)
    
    fig = plt.figure()
    lw = 2
    plt.plot(fpr_lr, recall_lr, color="tab:orange", lw=lw, 
             label="LR - AUC = %0.2f" % roc_auc_lr)
    plt.plot(fpr_rf, recall_rf, color="tab:blue", lw=lw, 
             label="RF - AUC = %0.2f" % roc_auc_rf)
    plt.plot(fpr_rf_cvx, recall_rf_cvx, color="tab:green", lw=lw, 
             label="RF cvx - AUC = %0.2f" % roc_auc_rf_cvx)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc="lower right")
    plt.show()
    return fig   


def performance_diagram(X_train, X_train_norm, X_train_cvx, y_train,
                        X_valid, X_valid_norm, X_valid_cvx, y_valid,
                        X_test, X_test_norm, X_test_cvx, y_test, repeats=10, **params):
    threshold = np.linspace(1e-8, 1, 100)
    recall_valid_rf = np.zeros((len(threshold),repeats))
    pres_valid_rf = np.zeros((len(threshold),repeats))
    recall_test_rf = np.zeros((len(threshold),repeats))
    pres_test_rf = np.zeros((len(threshold),repeats))
    recall_valid_rf_cvx = np.zeros((len(threshold),repeats))
    pres_valid_rf_cvx = np.zeros((len(threshold),repeats))
    recall_test_rf_cvx = np.zeros((len(threshold),repeats))
    pres_test_rf_cvx = np.zeros((len(threshold),repeats))
    recall_valid_lr_cvx = np.zeros((len(threshold),repeats))
    pres_valid_lr_cvx = np.zeros((len(threshold),repeats))
    recall_test_lr_cvx = np.zeros((len(threshold),repeats))
    pres_test_lr_cvx = np.zeros((len(threshold),repeats))
    for r in range(repeats):
        rf = create_RF(**params)
        rf.fit(X_train.values, y_train)
        y_valid_pred_proba_rf = rf.predict_proba(X_valid.values)[:,1]
        y_test_pred_proba_rf = rf.predict_proba(X_test.values)[:,1]
        rf_cvx = create_RF(**params)
        rf_cvx.fit(X_train_cvx.values, y_train)
        y_valid_pred_proba_rf_cvx = rf_cvx.predict_proba(X_valid_cvx.values)[:,1]
        y_test_pred_proba_rf_cvx = rf_cvx.predict_proba(X_test_cvx.values)[:,1]
        lr_cvx = create_LR()
        lr_cvx.fit(X_train_norm, y_train)
        y_valid_pred_proba_lr_cvx = lr_cvx.predict_proba(X_valid_norm)[:,1]
        y_test_pred_proba_lr_cvx = lr_cvx.predict_proba(X_test_norm)[:,1]
        for i in range(len(threshold)):
            y_test_pred_rf = np.where(y_test_pred_proba_rf > threshold[i], 1, 0)
            y_valid_pred_rf = np.where(y_valid_pred_proba_rf > threshold[i], 1, 0)
            recall_test_rf[i,r] = recall_score(y_test, y_test_pred_rf)
            pres_test_rf[i,r] = precision_score(y_test, y_test_pred_rf)
            recall_valid_rf[i,r] = recall_score(y_valid, y_valid_pred_rf)
            pres_valid_rf[i,r] = precision_score(y_valid, y_valid_pred_rf)
            #
            y_test_pred_rf_cvx = np.where(y_test_pred_proba_rf_cvx > threshold[i], 1, 0)
            y_valid_pred_rf_cvx = np.where(y_valid_pred_proba_rf_cvx > threshold[i], 1, 0)
            recall_test_rf_cvx[i,r] = recall_score(y_test, y_test_pred_rf_cvx)
            pres_test_rf_cvx[i,r] = precision_score(y_test, y_test_pred_rf_cvx)
            recall_valid_rf_cvx[i,r] = recall_score(y_valid, y_valid_pred_rf_cvx)
            pres_valid_rf_cvx[i,r] = precision_score(y_valid, y_valid_pred_rf_cvx)
            #
            y_valid_pred_lr_cvx = np.where(y_valid_pred_proba_lr_cvx > threshold[i], 1, 0)
            y_test_pred_lr_cvx = np.where(y_test_pred_proba_lr_cvx > threshold[i], 1, 0)
            recall_test_lr_cvx[i,r] = recall_score(y_test, y_test_pred_lr_cvx)
            pres_test_lr_cvx[i,r] = precision_score(y_test, y_test_pred_lr_cvx)
            recall_valid_lr_cvx[i,r] = recall_score(y_valid, y_valid_pred_lr_cvx)
            pres_valid_lr_cvx[i,r] = precision_score(y_valid, y_valid_pred_lr_cvx)
	# summarize results
    recall_rf_moy = np.mean(recall_test_rf, axis=1)
    pres_rf_moy, pres_rf_std = np.mean(pres_test_rf, axis=1), np.std(pres_test_rf, axis=1)
    recall_rf_cvx_moy = np.mean(recall_test_rf_cvx, axis=1)
    pres_rf_cvx_moy, pres_rf_cvx_std = np.mean(pres_test_rf_cvx, axis=1), np.std(pres_test_rf_cvx, axis=1)
    recall_lr_cvx_moy = np.mean(recall_test_lr_cvx, axis=1)
    pres_lr_cvx_moy, pres_lr_cvx_std = np.mean(pres_test_lr_cvx, axis=1), np.std(pres_test_lr_cvx, axis=1)
    
    recall_valid_rf_moy = np.mean(recall_test_rf, axis=1)
    pres_valid_rf_moy = np.mean(pres_test_rf, axis=1)
    recall_valid_rf_cvx_moy = np.mean(recall_test_rf_cvx, axis=1)
    pres_valid_rf_cvx_moy = np.mean(pres_test_rf_cvx, axis=1)
    recall_valid_lr_cvx_moy = np.mean(recall_test_lr_cvx, axis=1)
    pres_valid_lr_cvx_moy = np.mean(pres_test_lr_cvx, axis=1)
    
    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    xx, yy = np.meshgrid(x, y)
    csi_map = 1/(1/(xx) + 1/yy -1)
    csi_lr_cvx = 1/(1/(pres_valid_lr_cvx_moy) + 1/recall_valid_lr_cvx_moy -1)
    csi_rf = 1/(1/(pres_valid_rf_moy) + 1/recall_valid_rf_moy -1)
    csi_rf_cvx = 1/(1/(pres_valid_rf_cvx_moy) + 1/recall_valid_rf_cvx_moy -1)
    
    csi_lr_cvx_max = np.nanmax(csi_lr_cvx)
    csi_rf_max = np.nanmax(csi_rf)
    csi_rf_cvx_max = np.nanmax(csi_rf_cvx)
    
    index_max_lr_cvx = np.nanargmax(csi_lr_cvx)
    index_max_rf = np.nanargmax(csi_rf)
    index_max_rf_cvx = np.nanargmax(csi_rf_cvx)
    
    thr_lr_cvx = threshold[index_max_lr_cvx]
    thr_rf = threshold[index_max_rf]
    thr_rf_cvx = threshold[index_max_rf_cvx]

    fig, ax = plt.subplots()
    ct = ax.contourf(xx, yy, csi_map, 15, cmap='Blues')
    cbar = fig.colorbar(ct)
    cbar.set_label('CSI')
    ax.autoscale(False)
    plt.grid(b=True, which='major', linestyle='--')
    ax.fill_betweenx(recall_lr_cvx_moy, pres_lr_cvx_moy-pres_lr_cvx_std, pres_lr_cvx_moy+pres_lr_cvx_std, alpha=0.5, color='tab:orange')
    ax.plot(pres_lr_cvx_moy, recall_lr_cvx_moy, label='LR cvx - CSI = {:.2f}'.format(csi_lr_cvx_max), color='tab:orange')
    ax.scatter(pres_lr_cvx_moy[index_max_lr_cvx], recall_lr_cvx_moy[index_max_lr_cvx], marker='v', s=70, color='tab:orange')
    #
    ax.fill_betweenx(recall_rf_cvx_moy, pres_rf_cvx_moy-pres_rf_cvx_std, pres_rf_cvx_moy+pres_rf_cvx_std, alpha=0.5, color='tab:blue')
    ax.plot(pres_rf_cvx_moy, recall_rf_cvx_moy, label='RF - CSI = {:.2f}'.format(csi_rf_max), color='tab:blue')
    ax.scatter(pres_rf_cvx_moy[index_max_rf_cvx], recall_rf_cvx_moy[index_max_rf_cvx], marker='v', s=70, color='tab:blue')
    #
    ax.fill_betweenx(recall_rf_moy, pres_rf_moy-pres_rf_std, pres_rf_moy+pres_rf_std, alpha=0.5, color='tab:green')
    ax.plot(pres_rf_moy, recall_rf_moy, label='RF cvx - CSI = {:.2f}'.format(csi_rf_cvx_max), color='tab:green')
    ax.scatter(pres_rf_moy[index_max_rf], recall_rf_moy[index_max_rf], marker='v', s=70, color='tab:green')
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.legend(loc='lower left', framealpha=0.5)
    plt.minorticks_on()
    plt.axis([0,1.0,0,1.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return fig, thr_lr_cvx, thr_rf, thr_rf_cvx

def CSI_score(y_true, y_score):
    recall = recall_score(y_true, y_score)
    precision = precision_score(y_true, y_score)
    CSI = 1/(1/(precision) + 1/recall -1)
    return CSI
 

