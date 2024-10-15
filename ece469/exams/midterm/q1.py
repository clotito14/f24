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
from sklearn.model_selection import train_test_split

# import housing data from repository
url = "https://github.com/ageron/data/raw/main/housing/housing.csv"
housing = pd.read_csv(url)      # Store housing data in DataFrame
temp = housing

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
housing.hist(bins=100, color='skyblue', alpha=0.8, edgecolor='black')
plt.suptitle('California Housing Data Histogram')
plt.show()
plt.close()

# (d) Clean and normalize/standardize the data-set to make it appropriate
#     for training a regression model. Creating training and test sets.
#     Create a copy of the data with only the numerical attributes by
#     excluding the text attribute ocean proximity from the data-set.

# remove ocean_proximity from the dataset
housing = housing.drop('ocean_proximity', axis=1)

# clean data via imputer; replacing missing values with median or mean
simple_imputer = SimpleImputer(strategy='median')
housing_imputed = simple_imputer.fit_transform(housing)
housing = pd.DataFrame(housing_imputed, columns=housing.columns)

# standardize housing dataset
standard_scaler = StandardScaler()
housing_scaled = standard_scaler.fit_transform(housing)
housing = pd.DataFrame(housing_scaled, columns=housing.columns)
print('\nPREPROCESSED HOUSING DATA\n-------------------------')
print(housing)

# split into training and testing datasets
test_ratio = 0.2
housing_train, housing_test = train_test_split(housing, test_size=test_ratio, random_state=42)

print('\nHOUSING TRAINING SET\n--------------------')
print(housing_train)

print('\nHOUSING TESTING SET\n-------------------')
print(housing_test)


# (e) Because this data-set includes geographical information (latitude
#     and longitude), you are asked to create a scatterplot of all the
#     districts to visualize the geographical data in 2D space.

# get arrays containing lattitude and longitudes
lat = housing[['latitude']].values
long = housing[['longitude']].values

# plot them as a scatterplot
plt.scatter(lat, long, marker='o', s=0.75, c='#32a852', alpha=0.95, label='2D Map')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('2D Space Visualization of housing')
plt.legend()
plt.gca().set_facecolor((0,0.1,0.8, 0.1))
plt.show()
plt.close()

