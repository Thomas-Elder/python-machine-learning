#! python3

# preprocessing
# Let's see if we can do this before watching the video. 
# So: 

#%% Imports
import numpy
import matplotlib.pyplot as pyplot
import pandas

# SciKit imports
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

#%% Read data in 
dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

logging.debug('Value of X: %s' % (X))
logging.debug('Value of y: %s' % (y))

# Handle any blank cells (there aren't any but do anyway)

# Encode categorical data (there isn't any)
# Split into training and test set
# Scale sets