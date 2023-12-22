#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## print the graph after clustering
## label: start from 0 after clustering
## cents: Center of mass coordinates
## n_cluster: Number of clusters after clustering
## color: color of each cluster
def draw_result(train_x, label, cents, title):
    n_clusters = np.unique(labels).shape[0]
    color = ['red','orange','yellow']
    plt.figure()
    plt.title(title)
    for i in range(n_clusters):
        current_data = train_x[labels == i]
        plt.scatter(current_data[:,0], current_data[:,1], c = color[i])
        ## Use blue stars to indicate the center point position
        plt.scatter(cents[i, 0], cents[i, 1], c = 'blue', marker = '*', s = 100)
    return plt

if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_x = iris.data
    ## set the cluster num is 3
    clf = KMeans(n_clusters = 3, max_iter = 10, n_init = 10, init = 'k-means++', algorithm = 'full', tol = 1e-4, n_jobs = -1, random_state = 1)
    clf.fit(iris.x)
    print("SSE = {0}".format(clf.inertia_))
    draw_result(iris_x, clf.labels_, clf.cluster_centers_, 'kmeans').show()
## output will TypeError: KMeans.__init__() got an unexpected keyword argument 'n_jobs'
## I checked the reason should be old edition. Might try it later.


# In[ ]:




