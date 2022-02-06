#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:05:54 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/plot')

#####

import shapefile
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap

from netCDF4 import Dataset
from shapely.ops import cascaded_union

plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

#####

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

### Data acquisition
data_home_path = '/data/aubet'
nc = Dataset('/data/monika_louise/TRT_data/topo_DEM_50M.nc', 'r')
var = nc.variables['DEM'][:,:]
var = np.where(var<0, 0, var)
radar_loc = pd.read_csv(data_home_path+'/others/radar_locations.csv', sep=';', header=0)
borders = shapefile.Reader('/data/monika_louise/Border_CH.shp')

# A, D, L, P, W
align_x = ['right', 'right', 'left', 'right', 'left']
align_y = ['bottom', 'bottom', 'top', 'top', 'bottom']

offset_x = [-10, -6, 0, 0, 2]
offset_y = [40, 0, -27, -55, 20]

#####


fig = plt.figure()
cmap = plt.cm.get_cmap('viridis')
cmap_rev = cmap.reversed()
cmap_grey = grayscale_cmap(cmap_rev)
ax = fig.add_axes([0, 0, 1, 1])
alt = plt.imshow(np.flipud(var), alpha=.9, extent=[o_x,o_x+710,o_y,o_y+640],\
                 cmap=cmap_grey)
listx=[]
listy=[]
for shape in borders.shapeRecords():
    x = [i[0]*0.001 for i in shape.shape.points[:]]
    y = [i[1]*0.001 for i in shape.shape.points[:]]
    listx.append(x)
    listy.append(y)
    plt.plot(x, y, 'k', linewidth=1)
for i in range(len(radar_loc)):
    plt.scatter(radar_loc['X'].loc[i], radar_loc['Y'].loc[i], marker='.', color='k', s=50)             
    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=offset_x[i], y=offset_y[i], units='dots')
    plt.text(radar_loc['X'].loc[i], radar_loc['Y'].loc[i],
             radar_loc['Name'].loc[i]+'\n'+str(radar_loc['Z'].loc[i])+' m',
             horizontalalignment=align_x[i],
             verticalalignment=align_y[i],
             size='medium',transform=trans_offset)
plt.xlabel('Swiss X-coordinate [km]')
plt.ylabel('Swiss Y-coordinate [km]')
cbar = fig.colorbar(alt)
cbar.set_label('Altitude ASL [m]')
plt.show()
fig.savefig(code_home_path+'/outputs/map_CH.png', dpi=150, bbox_inches='tight')

#####

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
ax = fig.add_axes([0, 0, 1, 1])
alt = plt.imshow(var_new, alpha=.9,\
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
for i in range(len(radar_loc)):
    plt.scatter(radar_loc['X'].loc[i], radar_loc['Y'].loc[i], marker='.', color='k', s=50)             
plt.xlabel('Swiss X-coordinate [km]')
plt.ylabel('Swiss Y-coordinate [km]')
cbar = fig.colorbar(alt)
cbar.set_label('Altitude ASL [m]')
plt.show()


