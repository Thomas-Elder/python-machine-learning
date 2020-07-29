#! python3

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

# Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature scaling. Will need to transform the y set to a 2d array, as this 
# is what the scaler accepts as a parameter
y = y.reshape(len(y),1)

# We need separate scalers as they calculate the mean of the column to 
# scale the values. We want separate means for this.
standardScaler_X = StandardScaler()
standardScaler_y = StandardScaler()
X = standardScaler_X.fit_transform(X)
y = standardScaler_y.fit_transform(y)

# Training the svr model
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predict the result, then unscale the result 
y_pred = regressor.predict(standardScaler_X.transform([[6.5]]))
print(standardScaler_y.inverse_transform(y_pred)) # 170370.02

# Visualise the result


# Visualise the high res result
