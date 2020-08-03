#! python3

# Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)

def modelSelection_SupportVectorRegression(file: str):

    # Load data
    logging.debug('cwd: %s' % (os.getcwd()))
    dataset = pandas.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    logging.info('Len of X_train: \n{}'.format(len(X_train)))
    logging.info('Value of X_train: \n{}'.format(X_train))
    logging.info('Value of X_test: \n{}'.format(X_test))
    logging.info('Len of y_train: \n{}'.format(len(y_train)))
    logging.info('Value of y_train: \n{}'.format(y_train))
    logging.info('Value of y_test: \n{}'.format(y_test))

    # Feature scaling. Will need to transform the y set to a 2d array, as this 
    # is what the scaler accepts as a parameter
    #y_train = y_train.reshape(len(y),1)

    # We need separate scalers as they calculate the mean of the column to 
    # scale the values. We want separate means for this. X first.
    standardScaler_X_train = StandardScaler()
    standardScaler_X_train.fit(X)
    X_train_scaled = standardScaler_X_train.transform(X_train)
    X_test_scaled = standardScaler_X_train.transform(X_test)

    # Then y? Not sure here tbh
    #standardScaler_y_train = StandardScaler()
    #standardScaler_y_train.fit(y_train)
    #y_train_scaled = standardScaler_y_train.transform(y_train)

    logging.info('Len of X_train_scaled: \n{}'.format(len(X_train_scaled)))
    logging.info('Value of X_train_scaled: \n{}'.format(X_train_scaled))

    # Training the svr model
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train_scaled, y_train)

    # Predict the result, then unscale the result 
    y_pred = regressor.predict(X_test_scaled)

    return r2_score(y_test, y_pred)