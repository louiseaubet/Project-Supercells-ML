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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from helpers_preprocessing import split_with_IDs, prepare_sets, \
    normalisation
from helpers_model import create_ML_baseline, create_CNN
from helpers_graphs import plot_confusion_matrix

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#####

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

# Parameters
n_timesteps = 129
n_features = X_train.shape[2]
index_rank = feature_list.index("RANKr")


####################

# Create and train the models

### Grid search : Baseline CNN
model = KerasClassifier(build_fn=create_CNN_baseline, batch_size=50, verbose=0)
dropout_rate = [0.05, 0.1, 0.2]
neurons = [10, 20, 25, 50]
nb_filter = [8, 16, 32, 64]
optimizer = ['SGD', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(dropout_rate=dropout_rate, neurons=neurons, nb_filter=nb_filter,\
                  )
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1_weighted')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# Fit model
grid_result = grid.fit(X_train, y_train, batch_size=200, epochs=40,\
                      validation_data=(X_valid, y_valid), callbacks=[es])
print(grid_result.best_params_)
print(grid_result.best_score_)

#####

### Grid search : 1D CNN
model = KerasClassifier(build_fn=create_CNN, batch_size=200, verbose=0)
dropout_rate = [0.05, 0.1, 0.15, 0.2]
neurons = [100, 150, 200]
nb_filter = [32, 64, 128]
kernel_size = [5, 9, 11]

param_grid = dict(neurons=neurons)
    #, neurons=neurons, nb_filter=nb_filter, dropout_rate=dropout_rate
    #              kernel_size=kernel_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')

# Fit model
grid_result = grid.fit(X_train, y_train, batch_size=200, epochs=40,\
                      validation_data=(X_valid, y_valid))
print(grid_result.best_params_)
print(grid_result.best_score_)


#####
### Choose best model
best_CNN = create_CNN(0.15, 20)
history = best_CNN.fit(X_train, y_train, batch_size=10, verbose=2, epochs=10,
                    validation_data=(X_valid, y_valid))

#####
### Baseline
baseline = create_ML_baseline()
history = baseline.fit(X_train, y_train, batch_size=50, verbose=2, epochs=50,
                    validation_data=(X_valid, y_valid))

#################### 

### Make predictions
y_test_pred_proba_CNN = best_CNN.predict(X_test)
y_test_pred_CNN = np.where(y_test_pred_proba_CNN > 0.5, 1, 0)

fig = plot_confusion_matrix(y_test, y_test_pred_CNN)



