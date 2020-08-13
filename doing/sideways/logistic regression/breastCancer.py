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
X = dataset.iloc[:, 1:-1].values # The first column is Sample Code Number and we don't need it in the model
y = dataset.iloc[:, -1].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# create models
classifier_lr = LogisticRegression(random_state=0)
classifier_KNN = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier_SVM = SVC(kernel='linear', random_state=0)

# fit models
classifier_lr.fit(X_train, y_train)

# test model, the first record is [5,1,1,1,2,1,3,1,1,2], 
# so we can pass the model these features: [5,1,1,1,2,1,3,1,1], and expect that it returns 2.
logging.debug('classifier.predict([[5,1,1,1,2,1,3,1,1]]) == 2: {}'.format(classifier_lr.predict([[5,1,1,1,2,1,3,1,1]]) == 2)) # [ True]
y_pred = classifier_lr.predict(X_test)

# confusion matrix
cm_lr = confusion_matrix(y_test, y_pred)
logging.debug('cm_lr: {}'.format(cm_lr)) 
# [[87  0]
# [50  0]]
