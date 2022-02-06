#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:02:58 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/part_1')

#####

import pandas as pd
import numpy as np

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/raw/TRT_data_new/tstorm_regions.txt'
storm = pd.read_csv(data_path, sep=';', header=0, index_col=0)

#####

### Data cleaning

# Keep only relevant features
list_features_all = list(storm.columns)
list_features_ext = [name for name in list_features_all 
                     if ( (not name.startswith("p_")) & (not name.startswith("n_")) )]
storm = storm[list_features_ext]
storm = storm.drop(['RANK', 'region', 'pos', 'neg',
                    'mesostorm', 'hailstorm', 's_hailstorm', 'mesohailstorm', 
                    'meso_s_hailstorm', 'mesohail', 'meso_s_hail','normal',
                    'max_qual', 'min_qual', 'mean_qual', 'IQR_qual'], axis=1)

# Handling non numeric cells
storm = storm.replace('--', np.NaN)
storm = storm.astype(float)
storm['min_alt'].replace(-9999, 0, inplace=True);
col_area = [col for col in list_features_ext if 'area_' in col]
storm[col_area].replace(np.NaN, 0, inplace=True);
storm = storm.replace(np.NaN, -9999)

# Keeping data points with quality above 0.3
storm_qual = storm[storm['median_qual'] >= 0.3]

# One-hot encoding of region features
storm_dum = storm_qual
features = ['region_weighted', 'GWT8']
for i in range(len(features)):
    storm_dum = pd.concat([storm_dum,
                     pd.get_dummies(storm_qual[features[i]], prefix='cat_'+features[i])],
                      axis=1)
    storm_dum = storm_dum.drop([features[i]], axis=1)
    
storm_cvx = storm_dum

storm_rot = storm_dum.drop(['median_qual', 'ID', 'time'], axis=1) 

storm_rot.to_csv(data_home_path+'/processed/new/TRT_storm1_cleaned.txt', index=True)

##########

storm_rk = storm_rot.loc[storm_rot['RANKr'] >= 25]

storm_rk.to_csv(data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt', index=True) #_ID

##########

# Keeping only some features
storm_rot_ext = storm_rot
storm_rot_ext['vel'] = np.sqrt(storm_rot_ext['vel_x']**2 + storm_rot_ext['vel_y']**2)
list_features_to_keep = ['vel','area','det','RANKr','CG',
                         'ET45','ET15','maxH','VIL',
                         'max_CPC','median_POH','mean_MESHS','max_CZC',
                         'min_alt','max_slp','max_tpi',
                         'aspect','dalt','slope',
                         'dir_s','dis','meso']
list_features_all = list(storm_rot_ext.columns)
list_features_cat = [name for name in list_features_all if (name.startswith("cat_"))]
list_features_to_keep = list_features_to_keep + list_features_cat
storm_rot_ext = storm_rot_ext[list_features_to_keep]

storm_rot_ext.to_csv(data_home_path+'/processed/new/TRT_storm1_cleaned_extracted.txt', index=True)

##########

# Keeping only some features
storm_rk['vel'] = np.sqrt(storm_rk['vel_x']**2 + storm_rk['vel_y']**2)
list_features_to_keep = ['vel','area','det','RANKr','CG',
                         'ET45','ET15','maxH','VIL',
                         'max_CPC','median_POH','mean_MESHS','max_CZC',
                         'min_alt','max_slp','max_tpi',
                         'aspect','dalt','slope',
                         'dir_s','dis','meso']
list_features_all = list(storm_rk.columns)
list_features_cat = [name for name in list_features_all if (name.startswith("cat_"))]
list_features_to_keep = list_features_to_keep + list_features_cat
storm_rk_ext = storm_rk[list_features_to_keep]

storm_rk_ext.to_csv(data_home_path+'/processed/new/TRT_storm1_rk25_cleaned_extracted.txt', index=True)

##########

data_path_convexity = data_home_path+'/raw/convex_data.csv'
convexity_data = pd.read_csv(data_path_convexity, sep=',', header=0, index_col=0).reset_index()

storm_cvx = storm_cvx.loc[storm_cvx['RANKr'] >= 25]

storm_cvx = pd.merge(storm_cvx, convexity_data, on=['ID', 'time'], how ='left')
storm_cvx = storm_cvx.fillna(-9999)

storm_cvx = storm_cvx.drop(['ID', 'time', 'median_qual', 'hour'], axis=1)

storm_cvx.to_csv(data_home_path+'/processed/new/TRT_storm1_cvx_rk25_cleaned.txt', index=True)


