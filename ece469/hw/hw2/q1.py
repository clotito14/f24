###########################
# Chase Lotito - SIUC F24 #
# ECE469 - Intro to ML    #
# HW2 - Question 1        #
###########################

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in provided csv data to pandas dataframe
RAW_DATA_PATH = 'C:/Users/cloti/OneDrive/Desktop/CODE/datasets/datasetHW2P1.csv'
data = pd.read_csv(RAW_DATA_PATH)


# Split data into x and y vectors
x = data['x'].values        # inputs to non-linear system
y = data['y'].values        # outputs of non-linear system

# Data visualization (we see a curve like y=-x**3)
data.plot(kind='scatter', grid=True, x='x', y='y', title='Dataset HW2P1')
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)
plt.axis([xmin, xmax, ymin, ymax])
plt.show()

# (A) SPLIT DATASET TO CREATE TWO SUB DATASETS FOR TRAINING AND TESTING
