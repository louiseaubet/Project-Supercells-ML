#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 08:26:56 2022

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_1')

#####

import shapefile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
from helpers_model import data_preparation
from matplotlib.colors import LogNorm

plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

o_x=255
o_y=-160

#####

def grayscale_cmap(cmap):
    colors = cmap(np.arange(cmap.N))
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]  
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


#####

# Density of mesocyclones
data_home_path = '/data/aubet'
nc = Dataset('/data/monika_louise/TRT_data/topo_DEM_50M.nc', 'r')
var = nc.variables['DEM'][:,:]
var = np.where(var<0, 0, var)
radar_loc = pd.read_csv(data_home_path+'/others/radar_locations.csv', sep=';', header=0)
borders = shapefile.Reader('/data/monika_louise/Border_CH.shp')

quality = np.reshape(np.fromfile('/data/monika_louise/TRT_maps/vel_qual', dtype=np.float32),[640,710])
meso = np.reshape(np.fromfile('/data/monika_louise/TRT_maps/mtracks', dtype=np.float32),[640,710])
storm = np.reshape(np.fromfile('/data/monika_louise/TRT_maps/ttracks', dtype=np.float32),[640,710])

ratio = meso/storm # Ratio mesocyclones / thunderstorms

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

plt.rcParams.update({'font.size': 9})

fig = plt.figure()
cmap = plt.cm.get_cmap('viridis')
cmap_rev = cmap.reversed()
cmap_grey = grayscale_cmap(cmap_rev)
alt = plt.imshow(var_new, alpha=.9,\
                 extent=[o_x_new,o_x_new+dx_new,o_y_new,o_y_new+dy_new],\
                 cmap=cmap_grey)
listx=[]
listy=[]
meso = plt.imshow(np.flipud(ratio),\
                  norm=colors.LogNorm(vmin=0.004, vmax=0.3),\
                  extent=[o_x_new,o_x_new+dx_new,o_y_new,o_y_new+dy_new],\
                  cmap='viridis')
for shape in borders.shapeRecords():
    x = [i[0]*0.001 for i in shape.shape.points[:]]
    y = [i[1]*0.001 for i in shape.shape.points[:]]
    listx.append(x);listy.append(y)
    plt.plot(x, y, 'k', linewidth=0.7)
for i in range(len(radar_loc)):
    plt.scatter(radar_loc['X'].loc[i], radar_loc['Y'].loc[i], marker='.', color='k',\
                s=40)
plt.xlabel('Swiss X-coordinate [km]')
plt.ylabel('Swiss Y-coordinate [km]')
cbar = fig.colorbar(alt)
cbar.set_label('Altitude ASL [m]')
cbar.ax.tick_params(labelsize=9) 
cbar2 = fig.colorbar(meso)
cbar2.set_label('Ratio of mesocyclones [-]')
cbar2.ax.tick_params(labelsize=9)
plt.show()
fig.savefig(code_home_path+'/outputs/ratio_meso_nomeso.png', \
            dpi=150, bbox_inches='tight')

####

qual_arr = quality.flatten()
ratio_arr = ratio.flatten()

d = {'qual':qual_arr, 'ratio':ratio_arr}
df = pd.DataFrame(data=d)
df = df[df['qual'] >= 0]
df['bin_qual'] = pd.qcut(df['qual'], 35)
df_median = df.groupby(["bin_qual"]).median().reset_index()

fig = plt.figure()
plt.plot(df_median['qual'], df_median['ratio'], color='tab:blue')
plt.xlabel('Relative quality index [-]')
plt.ylabel('Ratio # mesocyclones / # thunderstorms [-]')
plt.grid(b=True, which='major', linestyle='--')
plt.minorticks_on()
plt.xlim([0, 1])
plt.ylim([0.013, 0.0285])
plt.show()
fig.savefig(code_home_path+'/outputs/part_1/threshold_ratio_qual_35bin.png', \
            dpi=150, layout='tight')

#####

### RANK density distribution

data_path = data_home_path+'/processed/new/TRT_storm1_cleaned_extracted.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

storm = storm.loc[storm['RANKr'] != 0]
meso = storm.loc[storm['meso'] == 1]
nomeso = storm.loc[storm['meso'] == 0]

fig, ax = plt.subplots()
plt.hist(nomeso['RANKr'], bins=40, density=True, alpha=0.6, color="tab:orange", \
         label='non-mesocyclones')
plt.hist(meso['RANKr'], bins=39, density=True, alpha=0.6, color="tab:blue", \
         label='mesocyclones')
plt.xlabel("RANK [-]")
plt.ylabel("Density distribution [-]")
plt.legend()
plt.minorticks_on()
plt.grid(True, linestyle='--')
plt.show()
fig.savefig(code_home_path+'/outputs/part_1/density_distributions_RANK_scale.png', \
            dpi=150, layout='tight')


### RANK density distribution with box plot
data_path = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt'
storm = pd.read_csv(data_path, sep=',', header=0, index_col=0)

fig, ax = plt.subplots()
plt = storm.boxplot('RANKr', by='meso')
plt.ylabel("RANK [-]")
plt.legend()
plt.minorticks_on()
plt.grid(True, linestyle='--')
plt.show()
fig.savefig(code_home_path+'/outputs/part_1/boxplot_RANK.png', \
            dpi=150, layout='tight')
