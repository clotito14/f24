"""
Southern Illinois University Carbondale
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    enriched_X = []
    y_temp = []
    for k in range(0, 4, 1):
        for i in range(0, size, 1):
            img = X[i].reshape(28,28)  # make image 28x28 matrix

            # image shifting logic
            if (k == 0):
                img = shift(img, [px, 0])  # shift up
            elif (k == 1):
                img = shift(img, [-px, 0]) # shift down
            elif (k == 2):
                img = shift(img, [0, px])  # shift right
            elif (k == 3):
                img = shift(img, [0, -px]) # shift left
            else:
                print('ERROR: image shift bounds error')

            img = img.flatten()     # make image 1D array
            enriched_X.append(img)
            y_temp.append(y[i])

    #enriched_X = np.array([img.flatten() for img in temp])
    enriched_Y = np.array(y_temp)
    return np.array(enriched_X), enriched_Y

# enrich X and y
e_X, e_y = quad_direction_enricher(X, y, len(X), px=1)

X = np.concatenate([X, e_X])
y = np.concatenate([y, e_y])


# scale down size of dataset
factor = 1
subset_size = 70000*factor
X = X[:subset_size]
y = y[:subset_size]

# split dataset into training and testing sets
test_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_ratio, random_state=42)

# next we want to standardize our data, but not
# targets, as to preserve y\in(0,9)
minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)

# train logisitic regression classifier

classify = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.1, max_iter=1000, random_state=42)
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
print(f'Probabilistic Model Tolerance     : {0.01}%')
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
n=3
knn = KNeighborsClassifier(n_neighbors=n, metric='minkowski')

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

# add KNN model results to finish classification report
print(f'Non-probabilistic Test      RMSE  : {test_rmse_knn:.2f}')
print(f'Non-probabilistic Test  Accuracy  : {test_accuracy_knn:.2f}%')
print(f'Non-probabilistic Test  F1 Score  : {test_f1_knn:.2f}')
print(f'Non-probabilistic Train     RMSE  : {train_rmse_knn:.2f}')
print(f'Non-probabilistic Train Accuracy  : {train_accuracy_knn:.2f}%')
print(f'Non-probabilistic Train F1 Score  : {train_f1_knn:.2f}')


# (D) Confusion Matrix
prob_cmatrix = confusion_matrix(y_pred=y_test_pred, y_true=y_test)
ConfusionMatrixDisplay(confusion_matrix=prob_cmatrix).plot(cmap='Blues')
plt.title('Confusion Matrix: Probabilistic Model')
plt.show()
plt.close()

nonprob_cmatrix = confusion_matrix(y_pred=y_test_pred_knn, y_true=y_test)
ConfusionMatrixDisplay(confusion_matrix=nonprob_cmatrix).plot(cmap='Reds')
plt.title('Confusion Matrix: Non-Probabilistic Model')
plt.show()
plt.close()
