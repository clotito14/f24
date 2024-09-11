# Chase Lotito - SIUC Fall 2024 - ECE469: Intro to Machine Learning
# HW1 - Question 3

import torch
import sklearn
from sklearn import preprocessing
import keras
import pandas as pd
import csv

RAW_DATA = './housing.csv'            # Full, untouched, housing data
PROCESSED_DATA = './processed.csv'    # Full, processed, housing data
TRAINING_DATA = './training.csv'      # Training housing data
TESTING_DATA = './testing.csv'        # Testing housing data

# read in the raw data from housing.csv into Pandas Dataframe
raw_data = pd.read_csv(RAW_DATA)

# convert the Pandas Datarframe to Numpy array
raw_data_array = raw_data.to_numpy()
print(raw_data_array[0][9])

# from here we need to ordinal encode oceanview
ord_enc = preprocessing.OrdinalEncoder().fit(raw_data_array)
enc_data_array = ord_enc.transform(raw_data_array)
print(enc_data_array)
