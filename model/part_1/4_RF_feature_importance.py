#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 08:22:13 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_1')

#####

import shap
shap.initjs()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from helpers_model import data_preparation, create_RF
from helpers_feature_importance import *

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

######

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

file_path = '/data/aubet/others/feature_list.csv'
explicit_features = pd.read_csv(file_path, sep=',', header=None)
explicit_features = list(explicit_features[0])

### Data preparation
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preparation(storm, 0.2, 0.2)

#######

params = {'n_estimators' : 200,
          'criterion' : 'entropy',
          'max_features' : 30,
          'max_depth' : 20}

rf = create_RF(**params)
rf.fit(X_train.values, y_train) 
y_pred_proba_rf = rf.predict_proba(X_test.values)[:,1]

explainer = shap.Explainer(rf)
choosen_instance = X_test
shap_values = explainer.shap_values(choosen_instance)

#######

# Feature importance plot
make_shap_waterfall_plot(shap_values[1], choosen_instance, explicit_features, 50)

#######

feature_groups = {
    "general":["time", "ID", "event", "year", "doy", "tod", "hour"],
    "intensity":["det","maxH","maxHm","RANKr",
                 "ET45","ET45m","ET15","ET15m",
                 "max_CZC", "mean_CZC", "median_CZC", "IQR_CZC"],
    "precipitation":["VIL", "hail", "s_hail", "POH",
            "max_POH", "mean_POH", "median_POH", "IQR_POH",
            "max_MESHS", "mean_MESHS", "median_MESHS", "IQR_MESHS",
            "max_CPC", "mean_CPC", "median_CPC", "IQR_CPC"],
    "lightning":["CG-","CG+","CG","%CG+"],
    "location":["lon", "lat", "chx", "chy", "cat_region_weighted_-9999.0",
                "cat_region_weighted_0.0", "cat_region_weighted_1.0",
                "cat_region_weighted_2.0", "cat_region_weighted_3.0",
                "cat_region_weighted_4.0", "cat_region_weighted_5.0",
                "cat_region_weighted_6.0", "cat_region_weighted_7.0",
                "cat_region_weighted_8.0" ],
    "topography":["altitude", "slope", "aspect", "dalt", "dslope", 
                 "max_alt", "mean_alt", "median_alt", "IQR_alt", "min_alt",
                 "max_slp", "mean_slp", "median_slp", "IQR_slp", "min_slp",
                 "max_asp", "mean_asp", "median_asp", "IQR_asp", "min_asp",
                 "max_tpi", "mean_tpi", "median_tpi", "IQR_tpi", "min_tpi"],
    "shape":["ell_L","ell_S","ell_or","area", "area_POH", "c_area_POH", "area_MESHS2",
             "c_area_MESHS2", "area_MESHS4", "c_area_MESHS4", "area_ref", "c_area_ref"],
    "motion":["vel_x","vel_y","Dvel_x","Dvel_y","dir","dir_s","dis"],
    "synoptic weather":["cat_GWT8_1.0", "cat_GWT8_2.0", "cat_GWT8_3.0", "cat_GWT8_4.0",
                        "cat_GWT8_5.0", "cat_GWT8_6.0","cat_GWT8_7.0","cat_GWT8_8.0",
                        "GWT10", "GWT18", "GWT26"]
    }


# Shap waterfall plot by group
flat_feature_groups = flatten_dict(feature_groups)
column_list = X_train.columns
shap_data = pd.DataFrame([column_list, np.abs(shap_values[1]).sum(0)]).T
shap_data = shap_data.rename(columns={0:'name', 1:'shap'})
shap_data['group'] = shap_data['name'].map(flat_feature_groups)

make_shap_waterfall_group_plot(shap_data)

#####

# Shap summary plot
fig_name = code_home_path+"/outputs/part_1/3_RF_interpretation/RF_feature_importance_summary.png"

shap.summary_plot(shap_values[1], choosen_instance, feature_names=explicit_features)

#####

def dependence_plot():
    fig = plt.figure()
    plt.axhline(y=0, linestyle='--', color='k')
    plt.scatter(, marker='.')
    plt.show()
    return

# Dependence plots
fig_path = code_home_path+'/outputs/part_1/3_RF_interpretation/feature_importance/'
top_features = ['area', 'min_tpi', 'RANKr', 'max_slp', 'det', 'area_MESHS4',\
                'mean_CZC']
explicit_names = ['Area [km²]', 'Minimum TPI [-]', 'RANK [-]',\
                  'Maximum slope [%]', 'Detection threshold [dBZ]',\
                  'Total area with MESHS > 4 cm [km²]',\
                  'Mean reflectivity [dBZ]']
feature_list = list(X_test.columns)
X_test = X_test.replace(-9999, np.NaN)



# Visualizing top features
for i in range(len(top_features)):
    feature = top_features[i]
    name = explicit_names[i]
    #
    fig_name = fig_path+"RF_dependence_"+feature+".png"
    index = feature_list.index(feature)
    x = X_test[feature].values
    y = shap_values[1][:,index]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    #
    fig = plt.figure()
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.axhline(y=0, linestyle='--', color='k')
    #plt.scatter(x, y, marker='.')
    kde = plt.scatter(x, y, c=z, s=20)
    plt.xlabel(name)
    plt.ylabel('SHAP value [-]')
    cbar = fig.colorbar(kde)
    cbar.set_label('Density of points [-]')
    plt.show()
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')


####

fig_path = code_home_path+"/outputs/part_1/3_RF_interpretation/"
top_features = ['max_CPC'] #, 'area_MESHS4','mean_CZC']
explicit_names = ['Maximum precipitation [mm/h]']
                 # 'Total area with MESHS > 4 cm [km²]',\
                 # 'Mean reflectivity [dBZ]']
feature_list = list(X_test.columns)
X_test = X_test.replace(-9999, np.NaN)

from scipy.stats import gaussian_kde

# Visualizing top features
for i in range(len(top_features)):
    feature = top_features[i]
    name = explicit_names[i]
    #
    fig_name = fig_path+"RF_dependence_"+feature+".png"
    index = feature_list.index(feature)
    x = X_test[feature].values
    y = shap_values[1][:,index]
    x = np.delete(x, 1740)
    y = np.delete(y, 1740)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    #
    fig = plt.figure()
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.axhline(y=0, linestyle='--', color='k')
    #plt.scatter(x, y, marker='.')
    kde = plt.scatter(x, y, c=z, s=20)
    plt.xlabel(name)
    plt.ylabel('SHAP value [-]')
    cbar = fig.colorbar(kde)
    cbar.set_label('Density of points [-]')
    plt.show()
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')
    


def scatter_hist(x, y, binx, biny, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, marker='.')
    ax.grid(b=True, which='major', linestyle='--')
    ax.minorticks_on()
    ax.axhline(y=0, linestyle='--', color='k')
    ax.set_ylabel('SHAP value [-]')

    # now determine nice limits by hand:
    binwidth_x = binx
    xmax = np.max(x)
    max_x = (int(xmax/binwidth_x) + 1) * binwidth_x
    xmin = np.min(x)
    min_x = (int(xmin/binwidth_x) + 1) * binwidth_x
    bins_x = np.arange(min_x - binwidth_x, max_x + binwidth_x, binwidth_x)
    
    binwidth_y = biny
    ymax = np.max(y)
    max_y = (int(ymax/binwidth_y) + 1) * binwidth_y
    ymin = np.min(y)
    min_y = (int(ymin/binwidth_y) + 1) * binwidth_y
    bins_y = np.arange(min_y - binwidth_y, max_y + binwidth_y, binwidth_y)
    
    ax_histx.hist(x, bins=bins_x)
    ax_histx.grid(b=True, which='major', linestyle='--')
    ax_histx.minorticks_on()
    ax_histx.set_ylabel('Number of points [#]')
    ax_histy.hist(y, bins=bins_y, orientation='horizontal')
    ax_histy.grid(b=True, which='major', linestyle='--')
    ax_histy.minorticks_on()
    ax_histy.set_xlabel('Number of points [#]')


# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]


fig_path = code_home_path+"/outputs/part_1/3_RF_interpretation/"
top_features = ['area', 'min_tpi', 'RANKr', 'max_slp', 'max_CPC']
explicit_names = ['Area [km²]', 'Minimum TPI [-]', 'RANK [-]',\
                  'Maximum slope [%]',\
                  'Maximum precipitation [mm/h]']
bins_x_arr = [50, 40, 1, 0.4, 10]
bins_y_arr = [0.01, 0.002, 0.002, 0.005, 0.001]
feature_list = list(X_test.columns)
X_test = X_test.replace(-9999, np.NaN)

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
for i in range(len(top_features)):
    feature = top_features[i]
    name = explicit_names[i]

    fig_name = fig_path+"RF_dependence_"+feature+"_hist.png"
    index = feature_list.index(feature)
    x = X_test[feature].values
    y = shap_values[1][:,index]
    x = np.delete(x, 1740)
    y = np.delete(y, 1740)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    scatter_hist(x, y, bins_x_arr[i], bins_y_arr[i], ax, ax_histx, ax_histy)
    ax.set_xlabel(name)
    plt.show()
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')
