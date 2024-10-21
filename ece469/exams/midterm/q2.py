"""
Southern Illinois University Carbonale
Department of Electrical Engineering
--------------------------------------
ECE469: Intro to Machine Learning
Midterm Exam: Question 2
Chase Lotito

10/15/2024

"MNIST"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score
from scipy.ndimage import shift

from sklearn.decomposition import PCA

# fetch the MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False)

# extract input features and output target
X = mnist['data']
y = mnist['target']

# the target values in y are strings, so we must
# first convert them to integers
y = y.astype(int)



# (b) Write a function that can shift an MNIST image
#     in any direction. Do this in all directions for 
#     the training set, and append them to it.
def quad_direction_enricher(X: np.array, y: np.array, size: int, px: int):
    """
    To enrich an image dataset with 4 sets of 
    the original set shifted in all directions px.
    (up, down, left, right)
    """
    temp = []
    y_temp = []
    for k in range(0, 4, 1):
        for i in range(0, size, 1):
            img = X[i].reshape(28,28)
            
            if (k == 0):
                img = shift(img, (px, 0))  # shift up
            elif (k == 1):
                img = shift(img, (-px, 0)) # shift down
            elif (k == 2):
                img = shift(img, (0, px))  # shift right
            elif (k == 3): 
                img = shift(img, (0, -px)) # shift left
            else:
                print('ERROR: image shift bounds error')

            temp.append(img)
            y_temp.append(y[i])

    #enriched_X = np.array([img.flatten() for img in temp])
    enriched_X = np.array([img.flatten() for img in temp])
    enriched_Y = np.array(y_temp)
    return enriched_X, enriched_Y

# enrich X
X, y = quad_direction_enricher(X, y, len(X), px=1)

print(f'len(X) = {len(X)}, len(y) = {len(y)}')

subset_size = 30000
# plot the examples of elements in the enriched dataset
#ax1 = plt.subplot(2, 4, 1)
#ax1.imshow(X[1].reshape(28,28))
#plt.title('Original NMIST')
#ax2 = plt.subplot(2, 4, 2)
#ax2.imshow(X[2].reshape(28,28))
#ax3 = plt.subplot(2, 4, 3)
#ax3.imshow(X[3].reshape(28,28))
#ax4 = plt.subplot(2, 4, 4)
#ax4.imshow(X[4].reshape(28,28))
#ax5 = plt.subplot(2, 4, 5)
#ax5.imshow(X[subset_size - 1].reshape(28,28))
#plt.title('Shifted NMIST', loc='center')
#ax6 = plt.subplot(2, 4, 6)
#ax6.imshow(X[2*subset_size - 2].reshape(28,28))
#ax7 = plt.subplot(2, 4, 7)
#ax7.imshow(X[3*subset_size - 3].reshape(28,28))
#ax8 = plt.subplot(2, 4, 8)
#ax8.imshow(X[4*subset_size - 4].reshape(28,28))
#plt.suptitle('Example Elements of Enriched MNIST')
#plt.show()
#plt.close()

# split dataset into training and testing sets
test_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_ratio, random_state=42)


# scale down size of dataset
test_size = int(subset_size*test_ratio)
train_size = int(subset_size*(1-test_ratio))
X_train = X_train[:train_size]
X_test = X_test[:test_size]
y_train = y_train[:train_size]
y_test = y_test[:test_size]

# next we want to standardize our data, but not
# targets, as to preserve y\in(0,9)
standard = StandardScaler()
X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

# train logisitic regression classifier

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_train = X_train_pca
X_test = X_test_pca

tolerance = 1e-3
classify = LogisticRegression(solver='lbfgs', penalty='l2', C=2, max_iter=1000, random_state=0)
classify.fit(X_train, y_train)

# predict using model
y_test_pred = classify.predict(X_test)
y_train_pred = classify.predict(X_train)

# calc MSE
test_rmse = root_mean_squared_error(y_test, y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)

# determine model accuracy
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
train_accuracy = accuracy_score(y_train, y_train_pred) * 100

# determine model f1 score
test_f1 = f1_score(y_test, y_test_pred, average='macro')
train_f1 = f1_score(y_train, y_train_pred, average='macro')

print('MNIST CLASSIFICATION REPORT')
print('###########################')
print(f'Dataset Size : {subset_size}')
print(f'Enriched Size : {len(X)}')
print(f'Probabilistic Model Tolerance     : {tolerance*100}%')
print(f'Probabilistic Test      RMSE      : {test_rmse:.2f}')
print(f'Probabilistic Test  Accuracy      : {test_accuracy:.2f}%')
print(f'Probabilistic Test  F1 Score      : {test_f1:.2f}')
print(f'Probabilistic Train     RMSE      : {train_rmse:.2f}')
print(f'Probabilistic Train Accuracy      : {train_accuracy:.2f}%')
print(f'Probabilistic Train F1 Score      : {train_f1:.2f}')

# (c) KNN-based algorithms belong to the class of non-probabilistic classifiers. 
#     You are asked to design a KNN-based classifier to classify handwritten 
#     digits (0-9) in MNIST data-set.

# initialize the knn classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

# train knn classifier
knn.fit(X_train, y_train)

# predict using knn classifier
y_test_pred_knn = knn.predict(X_test)
y_train_pred_knn = knn.predict(X_train)

# calc MSE
test_rmse_knn = root_mean_squared_error(y_test, y_test_pred_knn)
train_rmse_knn = root_mean_squared_error(y_train, y_train_pred_knn)

# determine model accuracy
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn) * 100
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn) * 100

# determine model f1 score
test_f1_knn = f1_score(y_test, y_test_pred_knn, average='macro')
train_f1_knn = f1_score(y_train, y_train_pred_knn, average='macro')

# add KNN model results to classification report
print(f'Non-probabilistic Test      RMSE  : {test_rmse_knn:.2f}')
print(f'Non-probabilistic Test  Accuracy  : {test_accuracy_knn:.2f}%')
print(f'Non-probabilistic Test  F1 Score  : {test_f1_knn:.2f}')
print(f'Non-probabilistic Train     RMSE  : {train_rmse_knn:.2f}')
print(f'Non-probabilistic Train Accuracy  : {train_accuracy_knn:.2f}%')
print(f'Non-probabilistic Train F1 Score  : {train_f1_knn:.2f}')