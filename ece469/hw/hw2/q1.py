###########################
# Chase Lotito - SIUC F24 #
# ECE469 - Intro to ML    #
# HW2 - Question 1        #
###########################

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Read in provided csv data to pandas dataframe
RAW_DATA_PATH = 'C:/Users/cloti/OneDrive/Desktop/CODE/datasets/datasetHW2P1.csv'
data = pd.read_csv(RAW_DATA_PATH)

test_size = 0.2


# (A) SPLIT DATASET TO CREATE TWO SUB DATASETS FOR TRAINING AND TESTING
x = data['x'].values
y = data['y'].values

x = x.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)




# (B) USE POLYNOMIAL REGRESSION TO FIT 5 POLYNOMIAL MODELS (DEG. 1 - DEG. 5)
deg = [1, 2, 3, 4, 5]
mse_train_arr = []
mse_test_arr = []

for i in deg:
    # transform input features into polynomial features
    poly = PolynomialFeatures(degree=i)    # initialize polynomial
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    # fit model
    poly_regressor = LinearRegression()
    poly_regressor.fit(x_train_poly, y_train)

    # test
    y_train_predicted = poly_regressor.predict(x_train_poly)
    y_test_predicted = poly_regressor.predict(x_test_poly)

    # calc mse
    mse_train = mean_squared_error(y_train, y_train_predicted)
    mse_test = mean_squared_error(y_test, y_test_predicted)

    # add mse's to arrays for later plotting
    mse_train_arr.append(mse_train)
    mse_test_arr.append(mse_test)

    # print mse for training and testing
    print(f"Degree: {i}")
    print(f"MSE Training: {mse_train}")
    print(f"MSE Testing: {mse_test}")

# plotting mse for training and testing against model complexity
plt.plot(deg, mse_train_arr, label='Training')
plt.plot(deg, mse_test_arr, label='Testing')
plt.xlabel('Model Complexity (Degree)')
plt.ylabel('Mean Square Error')
plt.title('Model Error v. Model Complexity')
plt.legend()
#plt.show()
#plt.close()




# (C) USE 10-FOLD CROSS-VALIDATION TO FIND THE MODEL WHICH OPTIMALLY FITS GIVEN DATASET.
#     PLOT THE TRAINING, CROSS-VALIDATION, AND TESTING ERRORS AGAINST MODEL COMPLEXITY.

# parameters
k = 10                 # for 10-fold cross-validation
best_degree = 1        # track best degree
max_degree = 5
best_mse = float('inf')
mse_per_degree = []

# perform k-fold cross-validation for each polynomial degree
for degree in range(1, max_degree + 1):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_fold_values = []
    k_iter = 1

    # initialize a plot for regression fit
    all_y_pred = np.zeros(x.shape)

    # Loop through each fold
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        # transform original x data into polynomial features for current degree
        poly = PolynomialFeatures(degree=degree)    # here degree is the parent loop iterator
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)

        # intialize and fit linear regression on polynomial space input features
        poly_regressor = LinearRegression()
        poly_regressor.fit(x_train_poly, y_train)         # <-- TRAINING MODEL HERE

        # predict using trained model
        y_test_pred = poly_regressor.predict(x_test_poly)

        # calc mse for test set
        mse_test = mean_squared_error(y_test, y_test_pred)
        mse_fold_values.append(mse_test)

    # calculuate avg mse for all folds for current deg
    avg_mse = np.mean(mse_fold_values)
    mse_per_degree.append(avg_mse)

    # check if current deg has lowest avg mse
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_degree = degree
    

    # Generate new plot for degree
    plt.figure(figsize=(12,8))
    plt.scatter(x,y,s=10,color='blue', label='Original Data', alpha=0.6)

    # plot polynomial fit for degree
    sorted_x = np.sort(x, axis=0)
    sorted_x_poly = poly.transform(sorted_x)
    plt.plot(sorted_x, poly_regressor.predict(sorted_x_poly), color='red', label=f"Degree {degree}", alpha = 0.7)

    # finalize plot for degree
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"poly_reg_deg{degree}_fold{k_iter}.png", dpi=300)
    plt.close()

# print best degree and mse
print(f"Best Polynomial Degree: {best_degree}")

# plot the mse v. polynomial degree
plt.figure(figsize=(10,6))
plt.plot([1,2,3,4,5], mse_per_degree, marker='o', color='b', label='Cross-Validation MSE', alpha=0.75)
plt.plot(deg, mse_train_arr, label='Training MSE', marker='+', alpha=0.75)
plt.plot(deg, mse_test_arr, label='Testing MSE', marker='x', alpha=0.75)
plt.title("MSE vs Model Complexity")
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(range(1, max_degree + 1))
plt.legend()
plt.grid(True)
plt.savefig('plots\\mse_vs_complexity.png', dpi=300)
plt.close()




# (D) CONSIDER A 4-DEGREE POLYNOMIAL AS YOUR MODEL.
#     USE RIDGE REGRESSION AND FIND BEST HYPERPARAMETER
#     LAMBDA VIA 10-F0LD CROSS-VALIDATION. PLOT THE CROSS
#     VALIDATION ERROR VERSUS LN(LAMBDA).

# remember we have x as input features and y as output features

# parameters
degree = 4                         # polynomial degree
lambdas = np.logspace(-4, 2, 100)  # lambdas for ridge
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# map input features into its polynomial space
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)

# store cross-validation results
mse_values = []

# perform cross-validation for each lambda
for alpha in lambdas:
    ridge = Ridge(alpha=alpha)

    # calc mse using cross_val_score
    mse = -cross_val_score(ridge, x_poly, y, cv=kf, scoring='neg_mean_squared_error').mean()
    mse_values.append(mse)

# find lambda with best minimum mse
best_lambda = lambdas[np.argmin(mse_values)]
print(f"Best lambda (w/ minimum MSE): {best_lambda}")

# fit model with best lambda
ridge_best = Ridge(alpha=best_lambda)
ridge_best.fit(x_poly, y)

# generate testing x values
x_fit = np.linspace(x.min(), x.max(), 1000).reshape(-1,1)
x_poly_fit = poly.transform(x_fit)   # send x_fit to polynomial space
y_fit = ridge_best.predict(x_poly_fit)

# plot mse vs. ln(lambda)
plt.figure(figsize=(8,6))
plt.plot(np.log(lambdas), mse_values, label='MSE')
plt.xlabel('$\\log_e (\\lambda)$')
plt.ylabel('Cross-Validated MSE')
plt.title('Cross-Validated MSE v. $\\log_e (\\lambda)$')
plt.grid(True)
plt.legend()
plt.savefig('crossvalmse_vs_loglambda.png', dpi=300)
plt.close()