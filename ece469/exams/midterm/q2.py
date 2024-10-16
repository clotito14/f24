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
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score
from scipy.ndimage import shift

# fetch the MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False)

# extract input features and output target
X = mnist['data']
y = mnist['target']

# the target values in y are strings, so we must
# first convert them to integers
y = y.astype(np.uint8)

subset_size = 70000


# (b) Write a function that can shift an MNIST image
#     in any direction. Do this in all directions for 
#     the training set, and append them to it.
def quad_direction_enricher(data: np.array, size: int, px: int) -> np.ndarray:
    """
    To enrich an image dataset with 4 sets of 
    the original set shifted in all directions px.
    (up, down, left, right)
    """
    temp = []
    for k in range(0, 4, 1):
        for i in range(0, size, 1):
            img = data[i].reshape(28, 28)
            
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

    enriched_data = temp

    return enriched_data

# enrich X
X = quad_direction_enricher(X, subset_size, px=1)

# enrich y, by elongating it by itself 4 times
y = np.concatenate((y,y,y,y), axis=0)

# scale down size of dataset
X_sub, y_sub = X[:subset_size*4], y[:subset_size*4]

# plot the examples of elements in the enriched dataset
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(X[1].reshape(28,28))
plt.title('Original NMIST')
ax2 = plt.subplot(2, 4, 2)
ax2.imshow(X[2].reshape(28,28))
ax3 = plt.subplot(2, 4, 3)
ax3.imshow(X[3].reshape(28,28))
ax4 = plt.subplot(2, 4, 4)
ax4.imshow(X[4].reshape(28,28))
ax5 = plt.subplot(2, 4, 5)
ax5.imshow(X[subset_size - 1].reshape(28,28))
plt.title('Shifted NMIST', loc='center')
ax6 = plt.subplot(2, 4, 6)
ax6.imshow(X[2*subset_size - 2].reshape(28,28))
ax7 = plt.subplot(2, 4, 7)
ax7.imshow(X[3*subset_size - 3].reshape(28,28))
ax8 = plt.subplot(2, 4, 8)
ax8.imshow(X[4*subset_size - 4].reshape(28,28))
plt.suptitle('Example Elements of Enriched MNIST')
plt.show()
plt.close()

# split dataset into training and testing sets
test_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=test_ratio, random_state=42)

# next we want to standardize our data, but not
# targets, as to preserve y\in(0,9)
standard = StandardScaler()
X_train = np.array(X_train).reshape(len(X_train), -1)
X_test = np.array(X_test).reshape(len(X_test), -1)
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

# train logisitic regression classifier
tolerance = 1e-4
classify = LogisticRegression(tol=tolerance, solver='lbfgs', max_iter=2200, random_state=42)
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
print(f'Model Tolerance : {tolerance*100}%')
print(f'Test      RMSE :  {test_rmse:.2f}')
print(f'Test  Accuracy :  {test_accuracy:.2f}%')
print(f'Test  F1 Score :  {test_f1:.2f}')
print(f'Train     RMSE :  {train_rmse:.2f}')
print(f'Train Accuracy :  {train_accuracy:.2f}%')
print(f'Train F1 Score :  {train_f1:.2f}')