# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 

import os
import logging

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Ok we're only selecting the last two columns, so that we're working with a 2D set we can plot.
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
# Basically look at the graph and where the angle of the wcss line flattens out is the point of diminishing returns.
wcss = []

# Now we compute the wcss for k clusters from k=1, 10.
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# We then plot these on a graph so we can see the turning point of fit. 
# If you have as many clusters as data points, you'll have wcss =0. This would be obvious 
# overfitting. With 1 cluster the wcss would be maximal. So it's about picking something
# in the middle
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
# We're using 5 as that looked good on the graph.
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# todo
# - run a regression of some sort on each column vs spending, to identify the most indicative feature
logging.debug('cwd: %s' % (os.getcwd()))
dataset = pandas.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)



# - run kmeans on other features to see other clusters?