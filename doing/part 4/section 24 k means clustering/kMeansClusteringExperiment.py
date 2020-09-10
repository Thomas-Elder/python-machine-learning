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
from sklearn.linear_model import LinearRegression

# Imports to encode the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import os
import logging

# - run a regression of some sort on each column vs spending, to identify the most indicative feature
# gender
print()
print(f'Column by column, firstly, gender')
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, -1].values

# So this turns X into a number of columns representing the different possible states, in this case, male or female.
# It drops the first column, to avoid the dummy variable trap. So here we end up with one column for Male, and if it's
# 1 it's male, if it's 0 it's female.
X = pd.get_dummies(data=X, drop_first=True)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train)
print(y_train)

# Regress
regressor = LinearRegression()
#RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Test
r2 = r2_score(y_test, y_pred)

print(f'r2: {r2}')

print(X)
print(y)

# Visualise
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Job Level - Simple Linear')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

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