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
    dataset = pandas.read_csv(file)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Will need to transform the y set to a 2d array, as this is what the scaler accepts as a parameter
    y = y.reshape(len(y),1)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Feature scaling. 
    # We need separate scalers as they calculate the mean of the column to 
    # scale the values. We want separate means for this. X first.
    standardScaler_X_train = StandardScaler()
    standardScaler_X_train.fit(X)
    X_train_scaled = standardScaler_X_train.transform(X_train)
    X_test_scaled = standardScaler_X_train.transform(X_test)

    # Then y? Not sure here tbh
    standardScaler_y_train = StandardScaler()
    standardScaler_y_train.fit(y_train)
    y_train_scaled = standardScaler_y_train.transform(y_train)

    # Training the svr model on the scaled training sets
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train_scaled, y_train_scaled)

    # Predict the result, then unscale the result 
    y_pred = standardScaler_y_train.inverse_transform(regressor.predict(X_test_scaled))

    return r2_score(y_test, y_pred)