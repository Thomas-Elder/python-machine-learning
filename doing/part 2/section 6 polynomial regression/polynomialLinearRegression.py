#! python3

# Ok let's see if we can get this one done before watching. 

# Imports
import numpy
import matplotlib.pyplot as plt
import pandas

# SciKit imports
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#from sklearn.PolynomialRegre import PolynomialRegre

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

# Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

# Encoding the categorical data, in column 1, the job
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = numpy.array(columnTransformer.fit_transform(X))

print(X)

# Split data into training and test set

# Regression

# Predict

# Test
