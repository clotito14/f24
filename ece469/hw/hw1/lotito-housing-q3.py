# Chase Lotito - SIUC Fall 2024 - ECE469: Intro to Machine Learning
# HW1 - Question 3

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder    # For encoding categorical features
from sklearn.impute import SimpleImputer            # For adding missing values
from sklearn.preprocessing import StandardScaler    # For standardizing data

# (A) Get housing data
RAW_DATA = 'https://github.com/ageron/data/raw/main/housing/housing.csv'
housing = pd.read_csv(RAW_DATA)

# (B) Choose input features and output features (saved into numpy.ndarray type)
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

# (C) Ocean Proximity is a categorical feature. Drop it or transform into numerical values (encode).

# Isolate the ocean_proximity data in input data X
ocean_proximity = X[:,8].reshape(-1,1)   # reshape(-1,1) to make 2D array for Ordinal
# Initalize the ordinal encoder
ordinal_encoder = OrdinalEncoder()
# Encode the ocean_proximity strings into numerical data
encoded_ocean = ordinal_encoder.fit_transform(ocean_proximity)
# Put the encoded version of ocean_proximity into input data X
X[:,8] = encoded_ocean.flatten()         # flatten to add 1D version of array back into X

# (D) Clean the dasta by either dropping or replacing missing values

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

# (E) Carry out feature scaling either via normalization or standardization.
std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(X)
X = scaled_data
scaled_data = std_scaler.fit_transform(Y)
Y = scaled_data

# (F) Create a training dataset and testing dataset
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# first, recombine X and Y into augmented matrix
housing_data = np.hstack((X,Y))

# split into testing and training set (both outputted as pd.DataFrames)
housing_training, housing_testing = shuffle_and_split_data(pd.DataFrame(housing_data), 0.2)
print('TRAINING:')
print(housing_training)
print('TESTING:')
print(housing_testing)