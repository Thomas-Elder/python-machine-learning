#! python3

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)

# Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values