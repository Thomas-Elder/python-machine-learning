#! python3

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

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

# Training the svr model
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predict the result
y_pred = regressor.predict([[6.5]])
print(y_pred) # 130001.82

# Visualise the result, it's a straight line!?
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Job Level - SVR - unscaled')
plt.xlabel('Level of Job')
plt.ylabel('Salary')
plt.show()

