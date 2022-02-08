#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:56:31 2022

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_2')

#####

import shapefile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from netCDF4 import Dataset

from helpers_preprocessing import split_with_IDs, prepare_sets, \
    random_oversampling, normalisation
from helpers_model import meteo_baseline, create_ML_baseline, create_CNN
from helpers_graphs import learning_curve, plot_confusion_matrix, F2_score_plot, \
    ROC_curve, precision_recall_curve, reliability_diagram, performance_diagram

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

o_x=350000
o_y=-50000

######

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

ratio_meso = y_train.sum()/(len(y_train))

# Parameters
n_timesteps = 129
n_features = X_train.shape[2]

model = create_CNN(nb_features=n_features, dropout_rate=0.05, nb_filter=128,\
                   neurons=200, kernel_size=7)
history_CNN = model.fit(X_train, y_train, epochs=40, batch_size=200, verbose=2,
                    validation_data=(X_valid, y_valid))
y_test_pred_proba_CNN = model.predict(X_test)

#######

# Spatial frequency
nc = Dataset('/data/monika_louise/TRT_data/topo_DEM_50M.nc', 'r')
var = nc.variables['DEM'][:,:]
radar_locations = pd.read_csv(data_home_path+'/others/radar_locations.csv', sep=';', header=0)

thr = 0.384
y_pred = np.where(y_test_pred_proba_CNN > thr, 1, 0) 
tp = (y_test * y_pred)
fn = (y_test * (y_pred==0))*2
fp = ((y_test == 0) * y_pred)*3
tn = ((y_test==0) * (y_pred==0))*4
output_type = tp+tn+fp+fn

spatial_freq = X_test[['ID', 'time', 'chx', 'chy']]
spatial_freq['type'] = output_type
spatial_freq_sorted = spatial_freq.sort_values(['ID', 'time'], ascending=True)

ID_list = spatial_freq['ID'].unique()

TP_arr = spatial_freq.loc[spatial_freq['type'] == 1]
FP_arr = spatial_freq.loc[spatial_freq['type'] == 3]
FN_arr = spatial_freq.loc[spatial_freq['type'] == 2]

borders = shapefile.Reader('/data/monika_louise/Border_CH.shp')
listx=[]
listy=[]
for shape in borders.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    listx.append(x); listy.append(y)


fig = plt.figure()
plt.imshow(np.flipud(var), alpha=.7, extent=[o_x,o_x+520000,o_y,o_y+450000], cmap='terrain')
for shape in borders.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    listx.append(x);listy.append(y)
    plt.plot(x, y, 'k', linewidth=1)
plt.scatter(TP_arr['chx'], TP_arr['chy'], color='g', label='TP', marker='.', s=10)
plt.scatter(FN_arr['chx'], FN_arr['chy'], color='b', label='FN', marker='.', s=10)
plt.scatter(FP_arr['chx'], FP_arr['chy'], color='r', label='FP', marker='.', s=10)
plt.scatter(radar_locations['X'], radar_locations['Y'], color='k', marker='.', s=50)
plt.xticks(rotation=45)
plt.xlabel('Swiss x-coordinates [m]')
plt.ylabel('Swiss y-coordinates [m]')
plt.legend(loc='upper left')
plt.show()
fig.savefig(code_home_path+'/outputs/part_2/1_CNN_interpretation/RF_spatial_frequency.png', \
            dpi=150, bbox_inches='tight')

