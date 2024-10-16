"""
Southern Illinois University Carbonale
Department of Electrical Engineering
--------------------------------------
ECE469: Intro to Machine Learning
Midterm Exam: Question 1
Chase Lotito

10/14/2024

"California Housing Prices"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.core.common import random_state
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# import housing data from repository
url = "https://github.com/ageron/data/raw/main/housing/housing.csv"
housing = pd.read_csv(url)      # Store housing data in DataFrame
temp = housing

# important variables
plots_url = "./plots/q1/"

# (a) Use info() method to identify the attributes of this data-set
print('HOUSING INFO\n-----------')
housing.info()


# (b) Use describe() method to identify and peek at a summary of
#     the numerical attributes.
housing_desc = housing.describe()
print('\nHOUSING DESCRIPTION\n-------------------')
print(housing_desc)


# (c) Use hist() method on the whole dataset and plot a histogram
#     for each numerical attribute. Notice that many histograms
#     are skewed right.
housing.hist(bins=100, color='skyblue', alpha=0.8, edgecolor='black', figsize=(12,8))
plt.suptitle('California Housing Data Histogram')
#plt.savefig(plots_url + 'housing_histogram.png', dpi=300)
plt.close()

# (d) Clean and normalize/standardize the data-set to make it appropriate
#     for training a regression model. Creating training and test sets.
#     Create a copy of the data with only the numerical attributes by
#     excluding the text attribute ocean proximity from the data-set.

# remove ocean_proximity from the dataset
housing = housing.drop('ocean_proximity', axis=1)

# clean data via imputer; replacing missing values with median or mean
# this satisfies part (i) since total_bedrooms now is complete
simple_imputer = SimpleImputer(strategy='median')
housing_imputed = simple_imputer.fit_transform(housing)
housing = pd.DataFrame(housing_imputed, columns=housing.columns)

#  STICKING PART H HERE SO I CAN ADD THESE BEFORE SPLITTING DATASET
#  (h) Add three new attributes; (i) rooms per house = total rooms/households, 
#     (ii) bedrooms ratio = total bedrooms/total rooms, and 
#     (iii) people per house = population/households.

# assign housing DataFrame attributes to arrays
total_rooms = housing['total_rooms'].values
households = housing['households'].values
total_bedrooms = housing['total_bedrooms'].values
population = housing['population'].values

# calculate new attributes
rooms_per_house = total_rooms / households
bedrooms_ratio = total_bedrooms / total_rooms
people_per_house = population / households

# assign new attributes to housing DataFrame
# using .insert(0, ...) to stack each in front
housing.insert(0, 'rooms_per_house', rooms_per_house)
housing.insert(0, 'bedrooms_ratio', bedrooms_ratio)
housing.insert(0, 'people_per_house', people_per_house)

# standardize housing dataset
standard_scaler = StandardScaler()
housing_scaled = standard_scaler.fit_transform(housing)
housing = pd.DataFrame(housing_scaled, columns=housing.columns)
print('\nPREPROCESSED HOUSING DATA\n-------------------------')
print(housing)

# extract input features and output target
x = housing[[
    'people_per_house',
    'bedrooms_ratio',
    'rooms_per_house',
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income'
    ]].values
y = housing['median_house_value'].values

# split into training and testing datasets
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=42)

# (e) Because this data-set includes geographical information (latitude
#     and longitude), you are asked to create a scatterplot of all the
#     districts to visualize the geographical data in 2D space.

# get arrays containing lattitude and longitudes
lat = housing[['latitude']].values
long = housing[['longitude']].values

# plot them as a scatterplot
plt.figure(figsize=(10,8))
plt.scatter(lat, long, marker='o', s=0.75, c='#32a852', alpha=0.95, label='2D Map')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title(r'Normalized 2D Space Visualization of $housing.csv$')
plt.legend()
plt.gca().set_facecolor((0,0.1,0.8, 0.1))
#plt.savefig(plots_url + '2d_housing_scatter.png', dpi=300)
plt.close()


# (f) Compute the standard correlation coefficient between every pair of
#     attributes using the corr() method.
corr_matrix = housing.corr()
corr_to_median_house_val = corr_matrix['median_house_value'].sort_values(ascending=False)
print('\nCORRELATION of ______ TO median_house_value\n-------------------------------------------')
print(corr_to_median_house_val)


# (g) Use scatter matrix (i.e., scatter_matrix() method) to plot every numerical 
#     attribute against every other numerical attribute, plus a histogram of each 
#     numerical attribute’s values on the main diagonal.
pd.plotting.scatter_matrix(housing, alpha=0.2, figsize=(20,16))
plt.suptitle('Scatter Matrix for $housing.csv$')
plt.tight_layout()
# plt.savefig(plots_url + 'housing_scatter_matrix.png', dpi=80)
plt.close()


# (i) Use the above preprocessed data-set to train a linear regression model to 
#     predict ”median house value”. Plot the training and test errors against the 
#     data-set size. Justify your results as much as possible.

# initalize linear model
linear_regressor = LinearRegression()

# train linear model
linear_regressor.fit(x_train, y_train)

# test linear model
y_train_predicted = linear_regressor.predict(x_train)
y_test_predicted = linear_regressor.predict(x_test)


# now test for different dataset sizes
train_sizes = np.linspace(0.1, 1.0, 10)  # Dataset sizes from 10% to 100% of the training set
train_errors = []
test_errors = []

for train_size in train_sizes:
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Use only a portion of the training set defined by `train_size`
    X_train_subset = X_train[:int(train_size * len(X_train))]
    y_train_subset = y_train[:int(train_size * len(y_train))]

    # Train the model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_subset, y_train_subset)

    # Make predictions
    y_train_pred = lin_reg.predict(X_train_subset)
    y_test_pred = lin_reg.predict(X_test)

    # Calculate the errors
    train_mse = mean_squared_error(y_train_subset, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Store the errors
    train_errors.append(train_mse)
    test_errors.append(test_mse)

# PLOTTING RESULTS

plt.figure(figsize=(16,9))

# plot predicted against actual
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(y_test, y_test_predicted, c='orange', marker='x', label='Learned Model', alpha=0.6)
ax1.plot([min(y_test), max(y_test)], [min(y_test_predicted), max(y_test_predicted)], c='blue', label='Ideal Model')
plt.xlabel('$\mathbf{y}_{test}$ Actual')
plt.ylabel('$\mathbf{y}_{test}$ Predicted')
plt.title('Learned Model Spread')
plt.legend()

## Plot the Learned Weights Coefficients of the model
#weights = linear_regressor.coef_
#feature_names = ['people_per_house', 'bedrooms_ratio', 'rooms_per_house','longitude', 
#                 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
#                 'population', 'households', 'median_income']
#
#ax2 = plt.subplot(1, 3, 2)
#ax2.barh(feature_names, weights)
#plt.xlabel("Model Weights $w_i$")
#plt.title("Linear Regression Feature Weights")

# Plot the learning curve
ax3 = plt.subplot(1, 2, 2)
ax3.plot(train_sizes, train_errors, label='Training Error', color='blue')
ax3.plot(train_sizes, test_errors, label='Test Error', color='green')
plt.title('Learning Curve')
plt.xlabel('Training Set Size (Ratio)')
plt.ylabel('Mean Squared Error')
plt.subplots_adjust(wspace=0.2)

plt.suptitle('Linear Model Results for $housing.csv$')
plt.legend()
plt.savefig(plots_url + "linear_model_results.png", dpi=300)
plt.close()