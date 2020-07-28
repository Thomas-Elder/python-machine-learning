#! python3

# Ok let's see if we can get this one done before watching. 

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

print(X)

# No encoding, we'll just ignore the position type and work with the level column.
# Train set (no splitting on this one)


# Regression, as a comparison we'll do a linear regression:
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# Regression, and the polynomical regression
# This polynomialFeatures creates additional features of increasingly higher order, here we've specified degree=2
# which means the expression looks like y = b + b1x1 + b2x1^2
# We then create a new LinearRegressor to take this new set of features.
polynomialFeatures = PolynomialFeatures(degree=2)
X_polynomial_2 = polynomialFeatures.fit_transform(X)
linearRegressor_2 = LinearRegression()
linearRegressor_2.fit(X_polynomial_2, y)

# And we can set up higher order polynomial expressions to closer fit the data
polynomialFeatures = PolynomialFeatures(degree=3)
X_polynomial_3 = polynomialFeatures.fit_transform(X)
linearRegressor_3 = LinearRegression()
linearRegressor_3.fit(X_polynomial_3, y)

polynomialFeatures = PolynomialFeatures(degree=4)
X_polynomial_4 = polynomialFeatures.fit_transform(X)
linearRegressor_4 = LinearRegression()
linearRegressor_4.fit(X_polynomial_4, y)

polynomialFeatures = PolynomialFeatures(degree=5)
X_polynomial_5 = polynomialFeatures.fit_transform(X)
linearRegressor_5 = LinearRegression()
linearRegressor_5.fit(X_polynomial_5, y)


# Visualise
plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor.predict(X), color = 'blue')
plt.title('Salary vs Job Level - Simple Linear')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor_2.predict(X_polynomial_2), color = 'blue')
plt.title('Salary vs Job Level - Polynomial Linear, degree 2')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor_3.predict(X_polynomial_3), color = 'blue')
plt.title('Salary vs Job Level - Polynomial Linear, degree 3')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor_4.predict(X_polynomial_4), color = 'blue')
plt.title('Salary vs Job Level - Polynomial Linear, degree 4')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor_5.predict(X_polynomial_5), color = 'blue')
plt.title('Salary vs Job Level - Polynomial Linear, degree 5')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()
