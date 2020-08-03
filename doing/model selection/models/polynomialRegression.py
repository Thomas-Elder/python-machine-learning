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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

def modelSelection_PolynomialRegression(file: str):
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

    # Train the model
    # Set up polynomial features
    poly = PolynomialFeatures(degree=3)

    # fit the features to the X_train set, and transform that set
    X_train_poly = poly.fit_transform(X_train)
    # use the train fit to transform the test set
    X_test_poly = poly.transform(X_test)

    # ok now set up a regressor and pass the poly features to that
    regressor = LinearRegression()
    regressor.fit(X_train_poly, y_train)

    # Predict, using the test poly set
    y_pred = regressor.predict(X_test_poly)

    return r2_score(y_test, y_pred)