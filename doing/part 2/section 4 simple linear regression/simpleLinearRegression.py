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

# Handle any blank cells (there aren't any)
# Encode categorical data (there isn't any)

#%% Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

logging.debug('Value of X_train: %s' % (X_train))
logging.debug('Value of X_test: %s' % (X_test))
logging.debug('Value of y_train: %s' % (y_train))
logging.debug('Value of y_test: %s' % (y_test))

# Scale sets
