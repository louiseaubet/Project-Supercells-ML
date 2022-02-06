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

import shap
shap.initjs()
import pandas as pd
import numpy as np
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from helpers_preprocessing import split_with_IDs, prepare_sets, normalisation
from helpers_interpretability import make_shap_waterfall_plot, flatten_dict, \
    make_shap_waterfall_group_plot

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

######

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_stormtracks_rk25_pad.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)
                    
### Data preparation
train, valid, test = split_with_IDs(storm, 0.2, 0.1)
train_n, valid_n, test_n = normalisation(train, valid, test)


### Prepare train and test sets                
# Parameters
n_timesteps = 129
X_train, y_train, feature_list = prepare_sets(train_n)
n_features = len(feature_list)
X_valid, y_valid, _ = prepare_sets(valid_n)
X_test, y_test, _ = prepare_sets(test_n)

X_test = np.delete(X_test, 11, axis=0)
y_test = np.delete(y_test, 11, axis=0)

file_path = data_home_path+'/others/feature_list_p2.csv'
explicit_features = pd.read_csv(file_path, sep=',', header=None)
explicit_features = list(explicit_features[0])

####################

dropout_rate = 0.15
nb_filter = 32
neurons = 200
kernel_size = 5
                   
inputs = keras.Input(shape=(n_timesteps, n_features))
x = keras.layers.Conv1D(filters=nb_filter, kernel_size=kernel_size, activation='relu', 
                 input_shape=(n_timesteps, n_features),
                 kernel_initializer='he_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.005))(inputs)
x = keras.layers.Dropout(dropout_rate)(x)
x = keras.layers.MaxPooling1D(pool_size=2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(neurons, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model_CNN = keras.Model(inputs=inputs, outputs=outputs, name="CNN_for_shap")
model_CNN.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])

history_CNN = model_CNN.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
y_test_pred_proba_CNN = model_CNN.predict(X_test)

background = X_train
explainer = shap.DeepExplainer(model_CNN, background)
choosen_instance = X_test
shap_values = explainer.shap_values(choosen_instance)

#######

# Feature importance plot
shap_values_mean = np.abs(shap_values[0]).mean(axis=1)
index_sorted = make_shap_waterfall_plot(shap_values_mean, feature_list, 20)

#####

# Shap summary plot
fig_name = code_home_path+"/outputs/part_2/1_CNN_interpretation/CNN_feature_importance_summary.png"
shap_values_mean = [shap_values[0].mean(axis=1)]
X_test_mean = pd.DataFrame(X_test.mean(axis=1), columns=feature_list)
shap.summary_plot(shap_values_mean[0], X_test_mean, feature_names=explicit_features)

#######

feature_groups = {
    "general":["time", "ID", "event", "year", "doy", "tod", "hour"],
    "intensity":["det","maxH","maxHm","RANKr",
                 "ET45","ET45m","ET15","ET15m",
                 "max_CZC", "mean_CZC", "median_CZC", "IQR_CZC"],
    "precipitation":["VIL", "hail", "s_hail",
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
                        "cat_GWT8_5.0", "cat_GWT8_6.0","cat_GWT8_7.0","cat_GWT8_8.0"]
    }


# Shap waterfall group plot
flat_feature_groups = flatten_dict(feature_groups)
shap_data = pd.DataFrame([feature_list, np.abs(shap_values_mean[0]).sum(0)]).T
shap_data = shap_data.rename(columns={0:'name', 1:'shap'})
shap_data['group'] = shap_data['name'].map(flat_feature_groups)

make_shap_waterfall_group_plot(shap_data)

#####

# Dependence plots

X_test_mean = pd.DataFrame(X_test.mean(axis=1), columns=feature_list)
X_test_not_norm, y_test, _ = prepare_sets(test)
X_test_not_norm = np.delete(X_test_not_norm, 11, axis=0)
X_test_not_norm_mean = pd.DataFrame(np.nanmean(X_test_not_norm,axis=1), columns=feature_list)
shap_values_mean = [shap_values[0].mean(axis=1)]

fig_path = code_home_path+"/outputs/part_2/1_CNN_interpretation/"
top_features = ['max_slp', 'area', 'median_CZC', 'min_tpi', 'det', 'Dvel_x']
explicit_names_norm = ['maximum slope [-]', 'area [-]', 'median reflectivity [-]',\
                  'minimum TPI [-]', 'detection threshold [-]', \
                  'change in x velocity [-]']
explicit_names = ['Maximum slope [%]', 'Area [km²]', 'Median reflectivity [dBZ]',\
                      'Minimum TPI [-]', 'Detection threshold [dBZ]', \
                      'Change in x velocity [km/h/15min]']
# Visualizing top features
for i in range(len(top_features)):
    feature = top_features[i]
    name = explicit_names_norm[i]
    #
    fig_name = fig_path+"CNN_dependence_"+feature+"_norm.png"
    index = feature_list.index(feature)
    shap = shap_values_mean[0]
    #
    fig = plt.figure()
    plt.grid(b=True, which='major', linestyle='--')
    plt.minorticks_on()
    plt.axhline(y=0, linestyle='--', color='k')
    plt.scatter(X_test_mean.iloc[:, index], shap[:,index], marker='.')
    plt.xlabel('Normalised '+name)
    plt.ylabel('SHAP value [-]')
    plt.show()
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')


####


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

X_test_mean = pd.DataFrame(X_test.mean(axis=1), columns=feature_list)
shap_values_mean = [shap_values[0].mean(axis=1)]
fig_path = code_home_path+"/outputs/part_2/1_CNN_interpretation/"
top_features = ['min_tpi', 'det', 'Dvel_x']
explicit_names_norm = ['minimum TPI [-]', 'detection threshold [-]', \
                  'change in x velocity [-]']
bins_x_arr = [0.02, 0.02, 0.001]
bins_y_arr = [0.00005, 0.00005, 0.0000005]

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
for i in range(len(top_features)):
    feature = top_features[i]
    name = 'Normalised '+explicit_names_norm[i]

    fig_name = fig_path+"CNN_dependence_"+feature+"_hist.png"
    index = feature_list.index(feature)
    x = X_test_mean.iloc[:, index]
    y = shap_values_mean[0][:,index]

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


#######

# Heatmap of feature importance
shap_values = explainer.shap_values(choosen_instance)
shap_values_grid = shap_values[0].mean(axis=0)
shap_values_grid_best = shap_values_grid[:,index_sorted[:20]]

fig = plt.figure()
plt.imshow(shap_values_grid, cmap='hot')
plt.xlabel('Timesteps')
plt.ylabel('Features')
plt.colorbar()
plt.show()
fig_name = code_home_path+"/outputs/part_2/1_CNN_interpretation/CNN_all_feature_importance_grid.png"
fig.savefig(fig_name, dpi=150, bbox_inches='tight')

fig = plt.figure()
plt.imshow(shap_values_grid_best, cmap='hot')
plt.xlabel('Timesteps')
plt.ylabel('Features')
plt.colorbar()
plt.show()
fig_name = code_home_path+"/outputs/part_2/1_CNN_interpretation/CNN_all_feature_importance_grid.png"
fig.savefig(fig_name, dpi=150, bbox_inches='tight')

