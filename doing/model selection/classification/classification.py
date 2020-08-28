
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

class Classification():

    def __init__(self, file: str):
        
        dataset = pd.read_csv(file)

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # scale
        self.sc = StandardScaler()
        self.X_train_scaled = self.sc.fit_transform(self.X_train)
        self.X_test_scaled = self.sc.transform(self.X_test)

        self.scores = []

    def train(self, classifier):

        # Train classifier
        classifier.fit(self.X_train_scaled, self.y_train)

        # Predict range
        y_pred = classifier.predict(self.X_test_scaled)

        acc = accuracy_score(self.y_test, y_pred)

        # if there are other measures we can add them here, print em all out later
        self.scores.append({'accuracy': acc})

        return round(acc*100,2)

