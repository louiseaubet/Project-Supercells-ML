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

from matplotlib.colors import LogNorm
from sklearn.metrics import recall_score, precision_score, auc, roc_curve
from sklearn.metrics import fbeta_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from helpers_model import create_ML_baseline, create_CNN

######

# Learning curve
def learning_curve(results):
    fig = plt.figure()
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    return fig
    
        
# Confusion matrix
def plot_confusion_matrix(y_test, y_test_pred, normalize=None):
    cf_matrix = confusion_matrix(y_test, y_test_pred, normalize=normalize)
    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, cbar=False, norm=LogNorm())
    plt.xlabel('Predicted class')
    plt.ylabel('Real class')
    plt.show()
    return fig

# F2_plot
def F2_score_plot(y_test, y_test_pred_proba_base, y_test_pred_proba):
    threshold = np.linspace(1e-8, 1, 100)
    F2_b = np.zeros((len(threshold),))
    F2 = np.zeros((len(threshold),))
    for i in range(len(threshold)):
        y_test_pred_base = np.where(y_test_pred_proba_base > threshold[i], 1, 0)
        F2_b[i] = fbeta_score(y_test, y_test_pred_base, beta=2)
        
        y_test_pred = np.where(y_test_pred_proba > threshold[i], 1, 0)
        F2[i] = fbeta_score(y_test, y_test_pred, beta=2)
    fig = plt.figure()
    plt.plot(threshold, F2_b, label='FFNN')
    plt.plot(threshold, F2, label='CNN')
    plt.xlabel('Decision threshold')
    plt.ylabel('F2-score')
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.axis('square')
    plt.legend()
    plt.show()
    #fig.savefig('/home/aubet/plots/part_1b/models/CNN_F2_score.png', \
    #        dpi=150, bbox_inches='tight')
    
# Precision-Recall curve
def precision_recall_curve(y_test, y_pred_proba_base, y_pred_proba):
    precision_b = precision_score(y_test, y_pred_proba_base)
    recall_b = recall_score(y_test, y_pred_proba_base)
    precision = precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred_proba)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    
    plt.figure()
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.plot(recall_b, precision_b, color="tab:blue", label="FFNN")
    plt.plot(recall, precision, color="darkorange", label="CNN")
    plt.plot([0, 1], [no_skill, no_skill], color="navy", linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.axis('square')
    plt.show()

    
# ROC curve
def ROC_curve(y_test, y_test_pred_proba_base, y_test_pred_proba):
    fpr_b, tpr_b, thresholds_b = roc_curve(y_test, y_test_pred_proba_base)
    roc_auc_b = auc(fpr_b, tpr_b)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr_b, tpr_b, color="tab:orange", lw=lw, 
             label="ANN - AUC = %0.2f" % roc_auc_b)
    plt.plot(fpr, tpr, color="tab:blue", lw=lw, 
             label="CNN - AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], color="k", lw=1, linestyle="--")
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FAR")
    plt.ylabel("Recall")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc="lower right")
    plt.show()
    return fig

def performance_meteo(y_train, y_rank_train, y_test, y_rank_test, thr_arr):
    recall_train_b = np.zeros((len(thr_arr),))
    pres_train_b = np.zeros((len(thr_arr),))
    recall_test_b = np.zeros((len(thr_arr),))
    pres_test_b = np.zeros((len(thr_arr),))
    for i in range(len(thr_arr)):
        y_train_pred = np.where(y_rank_train > thr_arr[i], 1, 0)
        recall_train_b[i] = recall_score(y_train, y_train_pred)
        pres_train_b[i] = precision_score(y_train, y_train_pred)
        #
        y_test_pred = np.where(y_rank_test > thr_arr[i], 1, 0)
        recall_test_b[i] = recall_score(y_test, y_test_pred)
        pres_test_b[i] = precision_score(y_test, y_test_pred)
    csi_train_b = 1/(1/(pres_train_b) + 1/recall_train_b -1)
    csi_train_b_max = np.nanmax(csi_train_b)
    index_max_b = np.nanargmax(csi_train_b)
    thr_b = thr_arr[index_max_b]
    #
    csi_test_b = 1/(1/(pres_test_b[index_max_b]) + 1/recall_test_b[index_max_b] -1)
    return thr_b, recall_test_b, pres_test_b, csi_test_b, index_max_b


def performance_diagram(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                        y_rank_train, y_rank_test, thr_arr, repeats=10):
    threshold = np.linspace(1e-8, 0.99999, 50)
    recall_valid_rf = np.zeros((len(threshold),repeats))
    pres_valid_rf = np.zeros((len(threshold),repeats))
    recall_valid_lr = np.zeros((len(threshold),repeats))
    pres_valid_lr = np.zeros((len(threshold),repeats))
    recall_test_rf = np.zeros((len(threshold),repeats))
    pres_test_rf = np.zeros((len(threshold),repeats))
    recall_test_lr = np.zeros((len(threshold),repeats))
    pres_test_lr = np.zeros((len(threshold),repeats))
    #
    thr_b, recall_b, precision_b, csi_b_test, index_max_b = performance_meteo(\
                y_train, y_rank_train, y_test, y_rank_test, thr_arr)
    #
    for r in range(repeats):
        rf = create_CNN()
        rf.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
        y_valid_pred_proba_rf = rf.predict(X_valid)
        y_test_pred_proba_rf = rf.predict(X_test)
        lr = create_ML_baseline()
        lr.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
        y_valid_pred_proba_lr = lr.predict(X_valid)
        y_test_pred_proba_lr = lr.predict(X_test)
        for i in range(len(threshold)):
            y_test_pred_rf = np.where(y_test_pred_proba_rf > threshold[i], 1, 0)
            y_valid_pred_rf = np.where(y_valid_pred_proba_rf > threshold[i], 1, 0)
            recall_test_rf[i,r] = recall_score(y_test, y_test_pred_rf)
            pres_test_rf[i,r] = precision_score(y_test, y_test_pred_rf)
            recall_valid_rf[i,r] = recall_score(y_valid, y_valid_pred_rf)
            pres_valid_rf[i,r] = precision_score(y_valid, y_valid_pred_rf)
            #
            y_valid_pred_lr = np.where(y_valid_pred_proba_lr > threshold[i], 1, 0)
            y_test_pred_lr = np.where(y_test_pred_proba_lr > threshold[i], 1, 0)
            recall_test_lr[i,r] = recall_score(y_test, y_test_pred_lr)
            pres_test_lr[i,r] = precision_score(y_test, y_test_pred_lr)
            recall_valid_lr[i,r] = recall_score(y_valid, y_valid_pred_lr)
            pres_valid_lr[i,r] = precision_score(y_valid, y_valid_pred_lr)
	# summarize results
    recall_rf_moy, recall_rf_std = np.mean(recall_test_rf, axis=1), np.std(recall_test_rf, axis=1)
    pres_rf_moy, pres_rf_std = np.mean(pres_test_rf, axis=1), np.std(pres_test_rf, axis=1)
    recall_lr_moy, recall_lr_std = np.mean(recall_test_lr, axis=1), np.std(recall_test_lr, axis=1)
    pres_lr_moy, pres_lr_std = np.mean(pres_test_lr, axis=1), np.std(pres_test_lr, axis=1)
    
    recall_valid_rf_moy = np.mean(recall_test_rf, axis=1)
    pres_valid_rf_moy = np.mean(pres_test_rf, axis=1)
    recall_valid_lr_moy = np.mean(recall_test_lr, axis=1)
    pres_valid_lr_moy = np.mean(pres_test_lr, axis=1)
    
    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    xx, yy = np.meshgrid(x, y)
    csi_map = 1/(1/(xx) + 1/yy -1)
    csi_lr = 1/(1/(pres_valid_lr_moy) + 1/recall_valid_lr_moy -1)
    csi_rf = 1/(1/(pres_valid_rf_moy) + 1/recall_valid_rf_moy -1)
    
    csi_lr_max = np.nanmax(csi_lr)
    csi_rf_max = np.nanmax(csi_rf)
    
    index_max_lr = np.nanargmax(csi_lr)
    index_max_rf = np.nanargmax(csi_rf)
    
    thr_lr = threshold[index_max_lr]
    thr_rf = threshold[index_max_rf]
    
    csi_lr_test = 1/(1/(pres_lr_moy[index_max_lr]) + 1/recall_lr_moy[index_max_lr] -1)
    csi_rf_test = 1/(1/(pres_rf_moy[index_max_rf]) + 1/recall_rf_moy[index_max_rf] -1)

    fig, ax = plt.subplots()
    ct = ax.contourf(xx, yy, csi_map, 15, cmap='Blues')
    cbar = fig.colorbar(ct)
    cbar.set_label('CSI')
    ax.autoscale(False)    
    plt.grid(b=True, which='major', linestyle='--')
    #
    ax.plot(precision_b, recall_b, label='Meteo baseline - CSI = {:.2f}'.format(csi_b_test), color='tab:green')
    ax.scatter(precision_b[index_max_b], recall_b[index_max_b], marker='v', s=70, color='tab:green')
    #
    ax.fill_betweenx(recall_lr_moy, pres_lr_moy-pres_lr_std, pres_lr_moy+pres_lr_std, alpha=0.5, color='tab:orange')
    ax.plot(pres_lr_moy, recall_lr_moy, label='ANN - CSI = {:.2f}'.format(csi_lr_test), color='tab:orange')
    ax.scatter(pres_lr_moy[index_max_lr], recall_lr_moy[index_max_lr], marker='v', s=70, color='tab:orange')
    #
    ax.fill_betweenx(recall_rf_moy, pres_rf_moy-pres_rf_std, pres_rf_moy+pres_rf_std, alpha=0.5, color='tab:blue')
    ax.plot(pres_rf_moy, recall_rf_moy, label='CNN - CSI = {:.2f}'.format(csi_rf_test), color='tab:blue')
    ax.scatter(pres_rf_moy[index_max_rf], recall_rf_moy[index_max_rf], marker='v', s=70, color='tab:blue')
    #
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.legend(loc='lower left', framealpha=0.5)
    plt.minorticks_on()
    plt.axis([0,1.0,0,1.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return fig, thr_b, thr_lr, thr_rf

def compute_freq(y_test, y_pred_proba, nb_bin=20):
    res = pd.DataFrame([y_test, y_pred_proba.flatten()]).T
    res = res.rename(columns={0:'true', 1:'proba_pred'})
    res['proba_pred_bin'] = pd.cut(res['proba_pred'], nb_bin)
    res['nb_pred'] = np.ones(y_test.shape)
    proba = res.groupby(["proba_pred_bin"])['true', 'nb_pred'].sum().reset_index()
    proba['rel_freq'] = proba['true']/proba['nb_pred']
    proba['forecast_proba'] = proba['proba_pred_bin'].apply(lambda x: x.mid).astype(float)
    clim = proba['true'].sum()/proba['nb_pred'].sum()
    return proba['forecast_proba'], proba['rel_freq'], proba['nb_pred'], clim

# Reliability diagram
def reliability_diagram(X_train, y_train, X_valid, y_valid, X_test, y_test, repeats=10):
    nb_bin = 20
    freq_rf = np.zeros((nb_bin, repeats))
    forecast_rf = np.zeros((nb_bin, repeats))
    freq_lr = np.zeros((nb_bin, repeats))
    forecast_lr = np.zeros((nb_bin, repeats))
    nb_rf = np.zeros((nb_bin, repeats))
    nb_lr = np.zeros((nb_bin, repeats))
    clim = np.zeros((repeats))
    for r in range(repeats):
        rf = create_CNN()
        rf.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
        y_valid_pred_proba_rf = rf.predict(X_valid)
        y_test_pred_proba_rf = rf.predict(X_test)
        lr = create_ML_baseline()
        lr.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
        y_valid_pred_proba_lr = lr.predict(X_valid)
        y_test_pred_proba_lr = lr.predict(X_test)
        forecast_rf[:,r], freq_rf[:,r], nb_rf[:,r], clim[r] = compute_freq(y_test, y_test_pred_proba_rf)
        forecast_lr[:,r], freq_lr[:,r], nb_lr[:,r], _ = compute_freq(y_test, y_test_pred_proba_lr)

	# summarize results
    forecast_rf_moy, forecast_rf_std = np.mean(forecast_rf, axis=1), np.std(forecast_rf, axis=1)
    freq_rf_moy, freq_rf_std = np.mean(freq_rf, axis=1), np.std(freq_rf, axis=1)
    nb_rf_moy = np.mean(nb_rf, axis=1)
    forecast_lr_moy, forecast_lr_std = np.mean(forecast_lr, axis=1), np.std(forecast_lr, axis=1)
    freq_lr_moy, freq_lr_std = np.mean(freq_lr, axis=1), np.std(freq_lr, axis=1)
    nb_lr_moy = np.mean(nb_lr, axis=1)
    clim_moy = np.mean(clim)

    SMALL_SIZE = 10
    line = np.linspace(0, 1, 100)
    array_one = np.ones((len(line),))
    fig = plt.figure()
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.plot(line, line, '--', lw=1, color='k')
    plt.axhline(y=clim_moy, lw=1, color='k', linestyle='-')
    #plt.axvline(x=clim_moy, lw=1, color='k', linestyle='-')
    sub_diag = 0.5*(clim_moy+line)
    #plt.plot(forecast_lr_moy, sub_diag, lw=1, color='k', linestyle='-')
    plt.fill_between(line, sub_diag, array_one, \
                     where=sub_diag>=clim_moy, alpha=0.3, color='tab:grey')
    plt.fill_between(line, sub_diag, \
                     where=sub_diag<=clim_moy, alpha=0.3, color='tab:grey')
    #
    plt.fill_between(forecast_lr_moy, freq_lr_moy-freq_lr_std, freq_lr_moy+freq_lr_std, alpha=0.5, color='tab:orange')
    plt.plot(forecast_lr_moy, freq_lr_moy, label='LR', lw=2, color='tab:orange')
    #
    plt.fill_between(forecast_rf_moy, freq_rf_moy-freq_lr_std, freq_rf_moy+freq_lr_std, alpha=0.5, color='tab:blue')
    plt.plot(forecast_rf_moy, freq_rf_moy, label='RF', lw=2, color='tab:blue')
    plt.grid(b=True, which='major', linestyle='--')
    plt.ylabel('Observed relative frequency')
    plt.xlabel('Forecast probability')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.axis([0,1,0,1])
    plt.gca().set_aspect('equal', adjustable='box')

    ax = plt.axes([0.32, 0.65, .17, .17])
    ax.plot(forecast_lr_moy, nb_lr_moy, color='tab:orange')
    ax.plot(forecast_rf_moy, nb_rf_moy, color='tab:blue')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(6)
    ax.set_ylabel('Frequency')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Forecast \n probability')

    plt.show()
    return fig


