
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

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)   # <-- train model here






