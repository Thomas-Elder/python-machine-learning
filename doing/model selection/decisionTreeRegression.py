#! python3

# Ok so same dataset, reckon I can just import same way as previous.

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)

# Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale? 

# Regress
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred)) # 0.9342783714449767