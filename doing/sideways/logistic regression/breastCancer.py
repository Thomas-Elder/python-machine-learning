#! python3

# Dataset from:
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# It's a set of data regarding breast cancer patients. 
# There are 10 features which predict whether the cancer is benign (2) or malignant (4)

# import libraries
import pandas as pd
import numpy as np

# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# import helpers
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, accuracy_score

# import misc
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)

# import dataset
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# create and train model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# test model, the first record is [1000025,5,1,1,1,2,1,3,1,1,2], 
# so we can pass the model these features: [1000025,5,1,1,1,2,1,3,1,1], and expect that it returns 2.
logging.debug('classifier.predict([[1000025,5,1,1,1,2,1,3,1,1]]) == 2: {}'.format(classifier.predict([[1000025,5,1,1,1,2,1,3,1,1]]) == 2)) # [ True]
y_pred = classifier.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
logging.debug('cm: {}'.format(cm)) 
# [[87  0]
# [50  0]]
