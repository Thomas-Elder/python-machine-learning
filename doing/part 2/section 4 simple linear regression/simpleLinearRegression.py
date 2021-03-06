#! python3

# preprocessing
# Let's see if we can do this before watching the video. 
# So: 

#%% Imports
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

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)

#%% Read data in 
dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

logging.info('Value of X: %s' % (X))
logging.info('Value of y: %s' % (y))

# Handle any blank cells (there aren't any)
# Encode categorical data (there isn't any)

#%% Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

logging.info('Value of X_train: %s' % (X_train))
logging.info('Value of X_test: %s' % (X_test))
logging.info('Value of y_train: %s' % (y_train))
logging.info('Value of y_test: %s' % (y_test))

# Scale feature sets (there's only one feature, so it's already in scale with itself)

# Train the simple linear regression model on the training set
# This creates and instance of LinearRegression class which uses ordinary least squares to 
# fit a model using the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%% Visualise training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#%% Visualise test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#%% Regression values
logging.info('y intercept: %s' %(regressor.intercept_))
logging.info('co-efficient: %s' %(regressor.coef_))

# With these values we can see the linear regression equation is:
# salary = 25609.89 + 9332.94 x experience
