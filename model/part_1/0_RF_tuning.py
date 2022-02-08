#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 07:53:32 2021

@author: aubet
"""

import os
code_home_path = '/home/aubet'
os.chdir(code_home_path+'/Project_Supercells_ML/model/part_1')

#####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from helpers_model import data_preparation

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

### Data acquisition
data_home_path = '/data/aubet'
data_path_1 = data_home_path+'/processed/new/TRT_storm1_cleaned_extracted.txt'
data_path_2 = data_home_path+'/processed/new/TRT_storm1_rk25_cleaned.txt'
storm = pd.read_csv(data_path_2, sep=',', header=0, index_col=0)

### Data preparation
X_train, X_valid, X_test, y_train, y_valid, y_test = data_preparation(storm, 0.2, 0.2)

model = RandomForestClassifier(n_estimators=50, criterion='entropy',
                            max_depth=20, random_state = 42,
                            max_features=8)

def grid_search(model):
    # Hyper-parameters
    n_estimators = [100, 150, 200]
    max_features = [20, 30, 40]
    max_depth = [20, 30, 40]

    param_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth}

    rf_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2,
                               n_jobs=-1, scoring='f1_weighted')
    return rf_cv

rf_cv = grid_search(model)     
rf_cv.fit(X_train, y_train)

rf_cv.best_params_
rf_cv.best_score_

best_model = RandomForestClassifier(**rf_cv.best_params_)
best_model.get_params

