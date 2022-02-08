#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:45:23 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_2')

#####

import pandas as pd
import numpy as np

# fix random seed for reproducibility
np.random.seed(42)

#####

### Data acquisition
data_home_path = '/data/aubet'
data_path = data_home_path+'/raw/TRT_data_new/tstorm_regions.txt'
storm = pd.read_csv(data_path, sep=';', header=0, index_col=0)

# data cleaning
storm['min_alt'].replace(-9999, 0, inplace=True);
storm = storm.replace('--', np.NaN)
storm = storm.replace(np.NaN, -9999)

# Keeping only relevant features
storm_ext = storm.drop(['hailstorm', 's_hailstorm', 'mesohailstorm', 'meso_s_hailstorm',
                        'meso', 'mesohail', 'meso_s_hail', 'region', 'POH',
                        'normal', 'pos', 'neg', 'RANK'], axis=1)
column_list = storm_ext.columns
rot_column = [name for name in column_list if (name.startswith("p_") or
                                                 name.startswith("n_"))]
storm_ext = storm_ext.drop(rot_column, axis=1)

# Converting hours to sinus values
storm_ext['hour_cos'] = np.cos(storm_ext['hour'])
storm_ext['hour_sin'] = np.sin(storm_ext['hour'])

storm_ext['tod_cos'] = np.cos(storm_ext['tod'])
storm_ext['tod_sin'] = np.sin(storm_ext['tod'])


# One-hot encoding of region features
storm_dum = storm_ext
features = ['region_weighted', 'GWT8']
for i in range(len(features)):
    storm_dum = pd.concat([storm_dum,
                     pd.get_dummies(storm_dum[features[i]], prefix='cat_'+features[i])],
                      axis=1)
    storm_dum = storm_dum.drop([features[i]], axis=1)

# Keep tracks with a lifecycle maximum RANK > 15
max_RANK = storm_dum.groupby('ID')['RANKr'].max().reset_index()
valid_IDs = max_RANK.loc[max_RANK['RANKr'] > 25]['ID']
storm_select = storm_dum.loc[storm['ID'].isin(valid_IDs)]

# Keep tracks with a lifecycle maximum relative quality > 0.3
max_qual = storm_select.groupby('ID')['median_qual'].max().reset_index()
valid_IDs = max_qual.loc[max_qual['median_qual'] >= 0.3]['ID']
storm_qual = storm_dum.loc[storm['ID'].isin(valid_IDs)]

storm_select = storm_qual.drop(['hour', 'tod', 'max_qual', 'min_qual', 'mean_qual',
                                'IQR_qual', 'median_qual'], axis=1)

# Compute maximum length of the tracks
size_tracks = storm_select.groupby('ID').size().reset_index().rename(columns={0: "size"})
max_size = int(size_tracks.max()[1])
mean_size = int(size_tracks.mean()[1])

# Pad all tracks to size max_size
ID_list = size_tracks['ID'].unique()
num_timestep = np.linspace(1, max_size, max_size)
list_tracks = []
for i in range(len(ID_list)):
    df = storm_select.loc[storm_select['ID'] == ID_list[i]].reset_index(drop=True)
    df_padded = df.reindex(range(max_size), fill_value=np.nan)
    df_padded['ID'] = ID_list[i]
    df_padded['num_timestep'] = num_timestep
    list_tracks.append(df_padded)

df_merged = pd.concat(list_tracks).reset_index(drop=True)

df_merged.to_csv(data_home_path+'/processed/new/TRT_storm1_stormtracks_rk25_pad.txt', index=True)
