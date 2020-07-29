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
logging.disable(logging.CRITICAL)

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

X_inv = standardScaler_X.inverse_transform(X)
y_inv = standardScaler_y.inverse_transform(y)

#%% Visualise the result
plt.scatter(X_inv, y_inv, color = 'red')
plt.plot(X_inv, standardScaler_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Salary vs Job Level - SVR')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

#%% Visualise the high res result
X_grid = numpy.arange(min(X_inv), max(X_inv), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_inv, y_inv, color = 'red')
plt.plot(X_grid, standardScaler_y.inverse_transform(regressor.predict(standardScaler_X.transform(X_grid))), color = 'blue')
plt.title('Salary vs Job Level - SVR - High Res')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# %%
