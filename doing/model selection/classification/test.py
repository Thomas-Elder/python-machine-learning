
from classification import Classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

classification = Classification('Data.csv')

models = []

models.append({'name': 'Logistic Regression', 'model': LogisticRegression(random_state=0)})
models.append({'name': 'Linear SVC', 'model': SVC(kernel='linear', random_state=0)})
models.append({'name': 'Kernel SVC', 'model': SVC(kernel='rbf', random_state=0)})
models.append({'name': 'KNN', 'model': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')})
models.append({'name': 'Naive Bayes', 'model': GaussianNB()})
models.append({'name': 'Decision Tree Classifier', 'model': DecisionTreeClassifier(criterion='entropy', random_state=0)})
models.append({'name': 'Random Forest Classifier', 'model': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)})

for model in models:
    result = classification.train(model['model'])
    print('Model {} had result: {}%'.format(model['name'], result))