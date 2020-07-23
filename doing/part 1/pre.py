#! python3

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

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

#%% Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values
dependantVariable = dataset.iloc[:, -1].values

#%% Check sets
logging.debug('Value of features: %s' % (features))
logging.debug('Value of dependantVariable: %s' % (dependantVariable))

#%% Handle empty cells
# missing_values: all occurrences of this will be imputed
# strategy: what will they be imputed (inferred) with? In this case the 'mean' 
# value for that column. 
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')

# Apply the imputation to the selected data from the set.
imputer.fit(features[:, 1:3])

# Transform the actual set with the imputed set.
features[:, 1:3] = imputer.transform(features[:, 1:3])

#%% Check sets
# The previously blank entries are now the mean value for their columns
logging.debug('Value of features: %s' % (features))
logging.debug('Value of dependantVariable: %s' % (dependantVariable))

#%% Encoding the categorical data, first the independant variables.
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
features = numpy.array(columnTransformer.fit_transform(features))

#%% Check sets
# The country data has been replaced by 3 columns, representing the 3 
# countries in that column. So column 1 is France, and has a 1 value in 
# the rows that contained the country France.
logging.debug('Value of features: %s' % (features))

#%% Encoding the categorical data, then the independent variables
labelEncoder = LabelEncoder()
dependantVariable = labelEncoder.fit_transform(dependantVariable)

#%% Check sets
# The purchase data has been replaced with 1 or 0 for yes/no
logging.debug('Value of dependantVariable: %s' % (dependantVariable))

#%% Split the dataset into training and test sets
# test_size is the % of data going in the test size, here 20%
features_train, features_test, dependantVariable_train, dependantVariable_test = train_test_split(features, dependantVariable, test_size = 0.2, random_state = 1)

#%% Check sets
# We now have 4 sets, features/dependantVariable train/test
logging.debug('Value of features_train: %s' % (features_train))
logging.debug('Value of dependantVariable_train: %s' % (dependantVariable_train))

logging.debug('Value of features_test: %s' % (features_test))
logging.debug('Value of dependantVariable_test: %s' % (dependantVariable_test))