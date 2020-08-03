#! python3

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

# Read data in 
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

logging.info('Value of X: \n{}'.format(X))
logging.info('Value of y: \n{}'.format(y))

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

logging.info('Value of X_train: \n{}'.format(X_train))
logging.info('Value of X_test: \n{}'.format(X_test))
logging.info('Value of y_train: \n{}'.format(y_train))
logging.info('Value of y_test: \n{}'.format(y_test))

# Train the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# So we don't have a million decimals
numpy.set_printoptions(precision=2)

# This just concatenates the y_pred and y_test arrays so the values can be compared. 
print(numpy.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print(r2_score(y_test, y_pred)) # 0.9321860060402446