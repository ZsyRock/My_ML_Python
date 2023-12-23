#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
print(os.path.abspath('.'))


# In[11]:


from sklearn import datasets
import numpy as np
from sklearn.neural_network import MLPClassifier
np.random.seed(0)
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
## slover is the weight optimization strategy; activation represents the selected activation function, there is no setting here, the default is relu; alpha is the penalty function; hidden_layer_sizes is the hidden layer size, the length is the number of hidden layers, each size is to set the neural network of each hidden layer The number of elements; random_state is the random direction used for initialization;
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
                   hidden_layer_sizes = (10, 10, 10), random_state = 1) ## if this hidden_layer_sizes = (5,2), the accuracy will be 0.2 only
clf.fit(iris_x_train, iris_y_train)
iris_y_predict = clf.predict(iris_x_test)
score = clf.score(iris_x_test, iris_y_test, sample_weight = None)

print(f'iris_y_predict = {iris_y_predict}')
print(f'iris_y_test = {iris_y_test}')
print(f'Accuracy: {score}')
print(f'layers nums: {clf.n_layers_}')


# In[ ]:




