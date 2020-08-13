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
from sklearn.model_selection import train_test_split, cross_val_score
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
classifier_KNN.fit(X_train, y_train)
classifier_SVM.fit(X_train, y_train)

# test models, the first record is [5,1,1,1,2,1,3,1,1,2], 
# so we can pass the model these features: [5,1,1,1,2,1,3,1,1], and expect that it returns 2.
logging.debug('classifier_lr.predict([[5,1,1,1,2,1,3,1,1]]) == 2: {}'.format(classifier_lr.predict([[5,1,1,1,2,1,3,1,1]]) == 2)) # [ True]
logging.debug('classifier_KNN.predict([[5,1,1,1,2,1,3,1,1]]) == 2: {}'.format(classifier_KNN.predict([[5,1,1,1,2,1,3,1,1]]) == 2)) # [ True]
logging.debug('classifier_SVM.predict([[5,1,1,1,2,1,3,1,1]]) == 2: {}'.format(classifier_SVM.predict([[5,1,1,1,2,1,3,1,1]]) == 2)) # [ True]

y_pred_lr = classifier_lr.predict(X_test)
y_pred_KNN = classifier_KNN.predict(X_test)
y_pred_SVM = classifier_SVM.predict(X_test)

# confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)

logging.debug('cm_lr: {}'.format(cm_lr)) 
# [[84  3]
# [3  47]]

logging.debug('cm_KNN: {}'.format(cm_KNN)) 
# [[84  3]
# [1  49]]

logging.debug('cm_SVM: {}'.format(cm_SVM)) 
# [[83  4]
# [2  48]]

logging.debug('Logistic Regression accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_lr)*100)) # 95.62%
logging.debug('K Nearest Neighbours accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_KNN)*100)) # 97.08%
logging.debug('Support Vector Machine accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred_SVM)*100)) # 95.62%

# Computing accuracy with k-fold cross validation
accuracies_lr = cross_val_score(estimator=classifier_lr, X=X_train, y=y_train, cv=10)
accuracies_KNN = cross_val_score(estimator=classifier_KNN, X=X_train, y=y_train, cv=10)
accuracies_SVM = cross_val_score(estimator=classifier_SVM, X=X_train, y=y_train, cv=10)

logging.debug('accuracies_lr.mean(): {:.2f}%'.format(accuracies_lr.mean()*100)) # 96.70%
logging.debug('accuracies_lr.std(): {:.2f}%'.format(accuracies_lr.std()*100)) # 1.97%

logging.debug('accuracies_KNN.mean(): {:.2f}%'.format(accuracies_KNN.mean()*100)) # 97.44%
logging.debug('accuracies_KNN.std(): {:.2f}%'.format(accuracies_KNN.std()*100)) # 1.85%

logging.debug('accuracies_SVM.mean(): {:.2f}%'.format(accuracies_SVM.mean()*100)) # 97.07%
logging.debug('accuracies_SVM.std(): {:.2f}%'.format(accuracies_SVM.std()*100)) # 2.19%