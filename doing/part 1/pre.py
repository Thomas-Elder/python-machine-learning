#! python3

#%% Imports
import numpy
import matplotlib.pyplot as pyplot
import pandas

import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#%% Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values
dependantVariable = dataset.iloc[:, -1].values

#%% Check sets
logging.debug('Value of features: %s' % (features))
logging.debug('Value of dependantVariable: %s' % (dependantVariable))