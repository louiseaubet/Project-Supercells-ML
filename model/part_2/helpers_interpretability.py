#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:42:33 2022

@author: aubet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers_model import create_ML_baseline, create_CNN

######

def make_shap_waterfall_plot(shap_values_mean, feature_list, num_display=20):
    column_list = np.array(feature_list)
    feature_ratio = (np.sum(np.abs(shap_values_mean), axis=0) / np.abs(shap_values_mean).sum()) * 100
    index_sorted = np.argsort(feature_ratio)[::-1].astype(int)
    column_list = column_list[index_sorted]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]
    
    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4
        
    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='navy', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], color='tab:blue', alpha=0.6)
    
    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio [%]')
    ax2.set_xlabel('Composition Ratio [%]')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))
    plt.show()
    fig_name = "/home/aubet/plots/part_2/post-analysis/CNN_20_feature_importance_waterfall.png"
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')
    return index_sorted

def flatten_dict(d):
    items = []
    for k, v in d.items():
        for i in range(len(v)):
            new_key = v[i]
            items.append((new_key, k))
    return dict(items)



def make_shap_waterfall_group_plot(data):
    feature_ratio = (data.groupby('group')['shap'].sum()/np.abs(data['shap']).sum())*100
    feature_ratio = feature_ratio.reset_index()
    feature_ratio_sorted = feature_ratio.sort_values(by='shap', ascending=False)
    feature_ratio_sorted['cum_sum'] = feature_ratio_sorted['shap'].cumsum(axis=0)
        
    fig, ax1 = plt.subplots()
    ax1.plot(feature_ratio_sorted['cum_sum'][::-1], feature_ratio_sorted['group'][::-1],\
             c='navy', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(feature_ratio_sorted['group'][::-1], feature_ratio_sorted['shap'][::-1],\
             color='tab:blue', alpha=0.6)
    
    ax1.grid(True, linestyle='--')
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(feature_ratio_sorted['cum_sum'].max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_sorted['shap'].max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio [%]')
    ax2.set_xlabel('Composition Ratio [%]')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(feature_ratio_sorted))
    #ax1.minorticks_on()
    #ax2.minorticks_on()
    plt.show()
    fig.savefig("/home/aubet/plots/part_2/post-analysis/CNN_20_feature_importance_waterfall_group.png", 
                dpi=150, bbox_inches='tight')

