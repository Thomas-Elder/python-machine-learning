#! python3

# Ok let's see if we can get this one done before watching. 
# The story goes this person is asking for 160k at a new job, and we want
# to figure out it that's reasonable based on their previous job. They were
# a Region Manager previously (150000) and had been so for 2 years.

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

# And just quickly, here's the desiredSalary to test against later. 
desiredSalary = 160000

# No encoding, we'll just ignore the position type and work with the level column.
# Train set (no splitting on this one)


# Regression, as a comparison we'll do a linear regression:
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# Regression, and the polynomical regression
# This polynomialFeatures creates additional features of increasingly higher order, here we've specified degree=2
# which means the expression looks like y = b + b1x1 + b2x1^2
# We then create a new LinearRegressor to take this new set of features.
polynomialFeatures_2 = PolynomialFeatures(degree=2)
X_polynomial_2 = polynomialFeatures_2.fit_transform(X)
linearRegressor_2 = LinearRegression()
linearRegressor_2.fit(X_polynomial_2, y)

# And we can set up higher order polynomial expressions to closer fit the data
polynomialFeatures_3 = PolynomialFeatures(degree=3)
X_polynomial_3 = polynomialFeatures_3.fit_transform(X)
linearRegressor_3 = LinearRegression()
linearRegressor_3.fit(X_polynomial_3, y)

polynomialFeatures_4 = PolynomialFeatures(degree=4)
X_polynomial_4 = polynomialFeatures_4.fit_transform(X)
linearRegressor_4 = LinearRegression()
linearRegressor_4.fit(X_polynomial_4, y)

polynomialFeatures_5 = PolynomialFeatures(degree=5)
X_polynomial_5 = polynomialFeatures_5.fit_transform(X)
linearRegressor_5 = LinearRegression()
linearRegressor_5.fit(X_polynomial_5, y)


#%% Visualise
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

#%% Predict salary with different models and compare results
y_pred_linear = linearRegressor.predict([[6.5]])
print(y_pred_linear) # $330378.78!!

y_pred_polynomial_2 = linearRegressor_2.predict(polynomialFeatures_2.fit_transform([[6.5]]))
print(y_pred_polynomial_2) # $189495.11, closer

y_pred_polynomial_3 = linearRegressor_3.predict(polynomialFeatures_3.fit_transform([[6.5]]))
print(y_pred_polynomial_3) # $133259.47, down?

y_pred_polynomial_4 = linearRegressor_4.predict(polynomialFeatures_4.fit_transform([[6.5]]))
print(y_pred_polynomial_4) # $158862.45, up?

y_pred_polynomial_5 = linearRegressor_5.predict(polynomialFeatures_5.fit_transform([[6.5]]))
print(y_pred_polynomial_5) # $174878.08, upper still!?  
