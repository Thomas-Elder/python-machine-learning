#! python3

#%% Imports
import numpy
import matplotlib.pyplot as pyplot
import pandas
from sklearn.impute import SimpleImputer

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