# Predicting rotation in supercell thunderstorms with supervised machine learning

### Description

This work aims to implement Random Forests and Convolutional Neural Networks
(CNN) to predict occurrences of supercells, based on radar-derived thunderstorm intensity
variables. They are trained with a multi-year dataset of supercell occurrences in Switzerland,
output from the operational Thunderstorm Radar Tracking algorithm and the Mesocyclone
Detection Algorithm (MDA) that identifies supercells among detected storm cells. A Random Forest is used to
predict the probability of rotation for each timestep and is compared to a Logistic Regression. A
CNN is used to predict the probability of rotation of storms, taking into account the entire track.
It is compared to a classic Artificial Neural Network (ANN) and a meteorological baseline that is
using only the RANK feature which evaluates the severity of the storm. These two models could
classify thunderstorms very fast and without the need for Doppler velocity data, which could
offer the opportunity to extend mesocyclone detection further back in time, where the capability
of the MDA algorithm may be limited.

### Getting Started

This version was designed for python 3.6.6 or higher. To run the model's calculation, use the Python Notebook `main.ipynb`. The parameters of the model can be set by the user in this file.


### Prerequisites

#### Libraries
The following librairies are used:
* [numpy](http://www.numpy.org/) 1.14.3, can be obtained through [anaconda](https://www.anaconda.com/download/)
* [pandas](https://pandas.pydata.org/), also available through anaconda
* [matplotlib](https://matplotlib.org/), also available through anaconda
* [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) 0.7.0 : `pip install imbalanced-learn`
* [scikit-learn](https://scikit-learn.org/stable/index.html) : `pip install -U scikit-learn`
* [plotly](https://plotly.com/python/) : `pip install plotly==4.14.1`
* [SHAP](https://shap.readthedocs.io/en/latest/index.html) : `pip install shap`
* [SHAP](https://pypi.org/project/pyshp/) : `pip install pyshp`
* [netCDF4](https://unidata.github.io/netcdf4-python/#quick-install) : `pip install netCDF4`

#### Code

The folder `model/part_1` contains the implementation of the Random Forest and the folder `model/part_2` contains the implementation of the CNN.
The folder `data` contains the input data used by the models.


### Additional content

The folder `outputs` contains the graphical results of this project.

The file `report.pdf` contains further informations on the background, the mathematical definition of the models as well as the analysis of the results.


### Authors

Aubet Louise, louise.aubet@epfl.ch


### Project Status

The project was submitted on the 7th of February 2022, as a master thesis at EPFL.
