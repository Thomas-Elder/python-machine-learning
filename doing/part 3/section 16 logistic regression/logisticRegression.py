#! python3

#%% Imports
import numpy
import matplotlib.pyplot as plt
import pandas

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Misc imports
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL)

# Load data
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scale
standardScaler = StandardScaler()
X_train_scaled = standardScaler.fit_transform(X_train)
X_test_scaled = standardScaler.transform(X_test)

# Model
model = LogisticRegression(random_state=0)
model.fit(X_train_scaled, y_train)

# Predict range
y_pred = model.predict(X_test_scaled)

# Predict single value
x_single = standardScaler.transform([[30, 87000]])
y_pred_single = model.predict(x_single)

# Compare results
numpy.set_printoptions(precision=2)
print(numpy.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print('y_pred_single:{}'.format(y_pred_single)) # predicted 0, no buy, correct

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[65  3] - 65 correct nobuy predictions, 3 incorrect
# [ 8 24]] - 8 incorrect buy predictions, 24 correct

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.89 - number of correct predictions divided by total number of predictions, so 89% correct.