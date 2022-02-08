#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:56:22 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_1')

#####

import pyefd
import pandas as pd
import numpy as np

from datetime import datetime
from csv import writer
from shapely.geometry import Polygon

def perimeter(contour):
    assert contour.ndim == 2 and contour.shape[1] == 2, contour.shape
 
    shift_contour = np.roll(contour, 1, axis=0)
    dists = np.sqrt(np.power((contour - shift_contour), 2).sum(axis=1))
    return dists.sum()

####################
    
data_home_path = '/data/aubet'
datapath = data_home_path+'/raw/TRT_extracted/'

storm_file = data_home_path+'/raw/TRT_data_new/tstorm_regions.txt'
storm = pd.read_csv(storm_file, sep=';', header=0, index_col=0)

# keep only timesteps that are contained in storm file
timesteps = storm['time'].unique().astype(int) # all timesteps in storm file
timesteps_new = np.zeros((len(timesteps),)).astype(str) # timesteps in correct format
filename_list = [] # names of the trt files
for n in range(len(timesteps)):
    date_time_obj = datetime.strptime(timesteps[n].astype(str), '%Y%m%d%H%M')
    timesteps_new[n] = date_time_obj.strftime('%y%j%H%M')
    filename_list.append('CZC'+timesteps_new[n].astype(str)+'0T.trt')

filename_list_small = [filename_list[x] for x in range(20)]

column_names = ['ID', 'time', 'convexity']
file_object = open(data_home_path+'/raw/convex_data.csv', 'a')
file_object.truncate(0)
writer_object = writer(file_object)
writer_object.writerow(column_names)

for filename in filename_list:
    try:
        text = pd.read_csv(datapath+filename)
        data = text.loc[8:]
        convexity = np.zeros((len(data),))
        
        for n in range(len(data)):
            t = data.iloc[n].str.split(';',expand=True)
            contour = np.array(t)[0,27:-1].astype(float)
            contour = np.reshape(contour,[int(len(contour)/2),2])
            
            # Get approximation by EFD
            efd_coeff = pyefd.elliptic_fourier_descriptors(contour, order=20, normalize=True)
            contour_efd = pyefd.reconstruct_contour(efd_coeff)

            p = Polygon(list(zip(contour_efd[:,1], contour_efd[:,0])))
            convex_hull = p.convex_hull
            x_convex, y_convex = convex_hull.exterior.coords.xy
            contour_convex = np.array([x_convex, y_convex]).T

            # Compute perimeter of the two contours
            p = perimeter(contour_efd)
            p_convex = perimeter(contour_convex)
            
            # Compute convexity
            convexity = p_convex/p
            
            id_info = t.iloc[:, 0:2].values.astype(int)
            
            new_line = np.append(id_info.astype(str), convexity.astype(str))
            writer_object.writerow(new_line)
            
    except:
        convexity_coeff = np.zeros((len(column_names),))
        convexity_coeff[:] = np.nan
        writer_object.writerow(convexity_coeff)
    
file_object.close()


