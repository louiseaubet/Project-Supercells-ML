#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:56:31 2022

@author: aubet
"""

import os
data_home_path = '/home/aubet'
os.chdir(data_home_path+'/Project_Supercells_ML/model/part_1')

#####

import shapefile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from netCDF4 import Dataset
from helpers_model import data_preparation, create_RF

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

o_x=350
o_y=-50

######

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned_ID.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

### Data preparation
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preparation(storm, 0.2, 0.2)

#######

params = {'n_estimators' : 200, # 50
          'criterion' : 'entropy',
          'max_features' : 30, # 10
          'max_depth' : 20,
          'random_state' : 42}

rf = create_RF(**params)
rf.fit(X_train.values, y_train) 
y_pred_proba_rf = rf.predict_proba(X_test.values)[:,1]

#######


# Spatial frequency
nc = Dataset('/data/monika_louise/TRT_data/topo_DEM_50M.nc', 'r')
var = nc.variables['DEM'][:,:]
radar_locations = pd.read_csv(data_home_path'/others/radar_locations.csv', sep=';', header=0)

thr = 0.3636
y_pred = np.where(y_pred_proba_rf > thr, 1, 0) 
tp = (y_test * y_pred)
fn = (y_test * (y_pred==0))*2
fp = ((y_test == 0) * y_pred)*3
tn = ((y_test==0) * (y_pred==0))*4
output_type = tp+tn+fp+fn

spatial_freq = X_test[['ID', 'time', 'chx', 'chy']]
spatial_freq['type'] = output_type
spatial_freq['chx'] = spatial_freq['chx']*0.001
spatial_freq['chy'] = spatial_freq['chy']*0.001
spatial_freq_sorted = spatial_freq.sort_values(['ID', 'time'], ascending=True)

ID_list = spatial_freq['ID'].unique()

TP_arr = spatial_freq.loc[spatial_freq['type'] == 1]
FP_arr = spatial_freq.loc[spatial_freq['type'] == 3]
FN_arr = spatial_freq.loc[spatial_freq['type'] == 2]

borders = shapefile.Reader('/data/monika_louise/Border_CH.shp')

o_x = 255
o_y = -160

dx = 710
dy = 640

o_x_new = 350
o_y_new = -50

dx_new = 520
dy_new = 450

n_x = var.shape[1]
n_y = var.shape[0]

l_x = dx/n_x
l_y = dy/n_y

a_x = int((o_x_new-o_x)/l_x)
a_y = int((o_y_new-o_y)/l_y)
 
d_x = a_x + int((dx_new)/l_x)
d_y = a_y + int((dy_new)/l_y)

var_new = np.flipud(var[a_y:d_y,a_x:d_x])

fig = plt.figure()
cmap = plt.cm.get_cmap('viridis')
cmap_rev = cmap.reversed()
cmap_grey = grayscale_cmap(cmap_rev)
alt = plt.imshow(var_new, alpha=.7,\
                 extent=[o_x_new,o_x_new+dx_new,o_y_new,o_y_new+dy_new],\
                 cmap=cmap_grey)
listx=[]
listy=[]
for shape in borders.shapeRecords():
    x = [i[0]*0.001 for i in shape.shape.points[:]]
    y = [i[1]*0.001 for i in shape.shape.points[:]]
    listx.append(x)
    listy.append(y)
    plt.plot(x, y, 'k', linewidth=1)
plt.scatter(TP_arr['chx'], TP_arr['chy'],color='tab:green',label='TP',marker='.', s=20)
plt.scatter(FN_arr['chx'], FN_arr['chy'],color='tab:blue',label='FN',marker='.', s=20)
plt.scatter(FP_arr['chx'], FP_arr['chy'],color='tab:orange',label='FP',marker='.', s=20)
plt.scatter(radar_locations['X'], radar_locations['Y'], color='k', marker='.', s=50)
plt.xlabel('Swiss x-coordinates [km]')
plt.ylabel('Swiss y-coordinates [km]')
plt.legend(loc='upper left')
cbar = fig.colorbar(alt)
cbar.set_label('Altitude ASL [m]')
plt.show()
fig.savefig(code_home_path+'/outputs/part_1/3_RF_interpretation/RF_spatial_frequency.png', \
            dpi=150, bbox_inches='tight')


######
# Correlation with minimum altitude visibility

filename = data_home_path+'/others/ami'
data = np.reshape(np.fromfile(filename, dtype=np.float32), [640,710])
data = np.flipud(data)

fig = plt.figure()
plt.imshow(data, alpha=.7, extent=[o_x,o_x+520000,o_y,o_y+450000], cmap='terrain')

x_arr = np.linspace(o_x, o_x+520000, 640)
y_arr = np.linspace(o_y, o_y+520000, 710)

def find_closest(x, x_arr):
    idx = (np.abs(x_arr - x)).argmin()
    return idx

spatial_freq['x_step'] = spatial_freq['chx'].apply(lambda col: find_closest(col, x_arr))
spatial_freq['y_step'] = spatial_freq['chy'].apply(lambda col: find_closest(col, y_arr))

spatial_freq['min_alt'] = data[spatial_freq['x_step'], spatial_freq['y_step']]

labels = ['TP', 'FN', 'FP', 'TN']
sns.boxplot(x="type", y="min_alt", data=spatial_freq)


