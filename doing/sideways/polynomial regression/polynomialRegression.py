#! python3

# Just wanted to go through this article: 
# https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

# Imports
import numpy as np
import operator
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

print('RMSE score of the linear regression is {}'.format(np.sqrt(mean_squared_error(y, y_pred)))) # 15.908242501429998
print('R2 score of linear regression is {}'.format(r2_score(y, y_pred))) # 0.6386750054827146

plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()

# This model underfits the data, to overcome under-fitting, we need to increase the complexity of the model.

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)

print('RMSE score of the polynomial linear regression is {}'.format(rmse)) # 10.120437473614711
print('R2 score of polynomial linear regression is {}'.format(r2)) # 0.8537647164420812

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()