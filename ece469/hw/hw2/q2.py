###########################
# Chase Lotito - SIUC F24 #
# ECE469 - Intro to ML    #
# HW2 - Question 2        #
###########################

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder    # For encoding categorical features
from sklearn.impute import SimpleImputer            # For adding missing values
from sklearn.preprocessing import StandardScaler    # For standardizing data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# (A) DOWNLOAD HOUSING.CSV

# Get housing data
RAW_DATA = 'https://github.com/ageron/data/raw/main/housing/housing.csv'
housing = pd.read_csv(RAW_DATA)



# (B) DATA-PREPROCESSING FROM HW1

# Choose input features and output features (saved into numpy.ndarray type)
X = housing[
        ['longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income',
        'ocean_proximity']
    ].values
Y = housing[['median_house_value']].values

# Ocean Proximity is a categorical feature. Drop it or transform into numerical values (encode).

# Isolate the ocean_proximity data in input data X
ocean_proximity = X[:,8].reshape(-1,1)   # reshape(-1,1) to make 2D array for Ordinal
# Initalize the ordinal encoder
ordinal_encoder = OrdinalEncoder()
# Encode the ocean_proximity strings into numerical data
encoded_ocean = ordinal_encoder.fit_transform(ocean_proximity)
# Put the encoded version of ocean_proximity into input data X
X[:,8] = encoded_ocean.flatten()         # flatten to add 1D version of array back into X

# Clean the dasta by either dropping or replacing missing values

# Initialized SimpleImputer, will use the median to add missing entries
simple_imputer = SimpleImputer(strategy='median')

# Change X np ndarray into a Pandas Dataframe to use SimpleImputer
dX = pd.DataFrame(X)
dY = pd.DataFrame(Y)

# Perform SimpleImputer transformation, for both inputs X and outputs Y
imputed_data = simple_imputer.fit_transform(dX)
X = imputed_data
imputed_data = simple_imputer.fit_transform(dY)
Y = imputed_data

# Carry out feature scaling either via normalization or standardization.
std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(X)
X = scaled_data
scaled_data = std_scaler.fit_transform(Y)
Y = scaled_data

# set test set size
test_size = 0.2

# split into testing and training set (both outputted as pd.DataFrames)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)




# (C) USE LINEAR REGRESSION TO DEVELOP AN ML MODEL FOR PREDICTION OF
#     'MEDIAN_HOUSE_VALUE' FOR FUTURE INPUTS AND ANALYZE TEST ERRORS
#      EXPLICITLY EXPRESS THE CORRESPONDING OPTIMAL WEIGHTS AND THE
#      FINAL LEARNED MODEL. USE GRAPHICAL REPRESENTATIONS.

# initalize and train linear model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)   # <-- train model here

# extract the optimal weights from the ML model
weights = linear_regressor.coef_.flatten()      # flatten makes it a normal list

# test model 
Y_train_predicted = linear_regressor.predict(X_train)
Y_test_predicted = linear_regressor.predict(X_test)

# calculate mean square error
mse_train = mean_squared_error(Y_train, Y_train_predicted)
mse_test = mean_squared_error(Y_test, Y_test_predicted)

# VISUALIZATION
# print out results of linear model
print("--------------------")
print("LINEAR MODEL RESULTS")
print("--------------------")
print(f"Model Weights: {weights}")
print(f"MSE (train): {mse_train*100:.2f}%")
print(f"MSE (test): {mse_test*100:.2f}%")

# Plot predicted vs actual values
plt.scatter(Y_test, Y_test_predicted, marker='o', s=0.75, c='#32a852', alpha=0.95, label='Pred v. Actual')     # plot predicted against actual, if diagonalized, well fit.
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], label='Ideal Model',color='red', linewidth=2)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Predicted Median House Value vs Actual Median House Value")
plt.legend()
plt.savefig("plots\\q2_pred_vs_actual.png", dpi=300)
plt.close()

# Plot the Learned Weights
# Coefficients of the model
#feature_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
#                 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
#
#plt.barh(feature_names, weights)
#plt.xlabel("Model Weights $w_i$")
#plt.title("Linear Regression Feature Weights")
#plt.show()
#plt.close()



# (D) USE CROSS-VALIDATION TECHNIQUES TO IMPROVE THE GENERALIZATION OF THE MODEL AND ANALYZE
#     THE ROOT MEAN SQUARE ERROR (RMSE). USE GRAPHICAL ILLUSTRATIONS.

# parameters
lambdas = np.logspace(-4, 2, 100)  # lambdas for ridge
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# store cross-validation results
mse_values = []

# perform cross-validation for each lambda
for alpha in lambdas:
    ridge = Ridge(alpha=alpha)

    # calc mse using cross_val_score
    mse = -cross_val_score(ridge, X, Y, cv=kf, scoring='neg_mean_squared_error').mean()
    mse_values.append(mse)

# find lambda with best minimum mse
best_lambda = lambdas[np.argmin(mse_values)]
print("--------------------")
print("RIDGE REGULARIZATION")
print("--------------------")
print(f"Best lambda (w/ minimum MSE): {best_lambda:.2f}")
print(f"Minimum MSE: {np.min(mse_values)*100:.2f}%")

# fit model with best lambda
ridge_best = Ridge(alpha=best_lambda)
ridge_best.fit(X_train, Y_train)

# testing x values
Y_test_ridge_pred = ridge_best.predict(X_test)

# plot rmse vs. ln(lambda)
plt.figure(figsize=(8,6))
plt.plot(np.log(lambdas), np.sqrt(mse_values), c="#570710",label='MSE')
plt.xlabel('$\\log_e (\\lambda)$')
plt.ylabel('Cross-Validated RMSE')
plt.title('Q2: Cross-Validated RMSE v. $\\log_e (\\lambda)$')
plt.grid(True)
plt.legend()
plt.savefig('plots\\q2_crossvalmse_vs_loglambda.png', dpi=300)
plt.close()