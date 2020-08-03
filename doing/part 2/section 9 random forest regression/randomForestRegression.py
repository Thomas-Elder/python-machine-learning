#! python3

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 

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

# Regress
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print('Prediction {}'.format(y_pred)) # Prediction [158300.]

# Visualise the result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Job Level - DTR')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

# High res.... ?!
X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Job Level - DTR - High Res')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()