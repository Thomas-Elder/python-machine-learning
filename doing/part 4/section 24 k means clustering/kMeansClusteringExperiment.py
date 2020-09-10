# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Imports to encode the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import os
import logging

# - run a regression of some sort on each column vs spending, to identify the most indicative feature
# sex
print()
print(f'Column by column, firstly, sex')
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Need to convert the sex column into categories... #todo
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X))

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Regress
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Test
r2 = r2_score(y_test, y_pred)

print(f'r2: {r2}')

# age
print()
print(f'Secondly, age')
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, -1].values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Regress
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Test
r2 = r2_score(y_test, y_pred)

print(f'r2: {r2}')

# income
print()
print(f'Thirdly, income')
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:4].values
y = dataset.iloc[:, -1].values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Regress
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Test
r2 = r2_score(y_test, y_pred)

print(f'r2: {r2}')

# - run kmeans on other features to see other clusters?