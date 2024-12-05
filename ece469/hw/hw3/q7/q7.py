'''
==========================================
Homework 3 Question 7 Code
Name       : Chase Lotito
University : Southern Illinois University
Course     : ECE469
==========================================
Description:
Feed-forward neural network
==========================================
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.io import loadmat

# import dataset
data = loadmat('../dataset1.mat')
X = data['X']
y = data['Y']

plt.scatter(X[y[:,0] == 0, 0], X[y[:,0] == 0, 1], color='red', label='Class 0')
plt.scatter(X[y[:,0] == 1, 0], X[y[:,0] == 1, 1], color='blue', label='Class 1')
plt.title('Scatterplot of dataset1.mat')
plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend()
plt.show()