import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

# Download and extract data
data_link = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_link + "lifesat/lifesat.csv")
X = life_sat[["GDP per capita (USD)"]].values                 # choose input vector
Y = life_sat[["Life satisfaction"]].values                    # choose output vector

# Data visualization
life_sat.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction', title='Satisfaction v. GDP')
plt.axis([25000, 62000, 5, 8])
plt.show()

# Model this dataset
model = LinearRegression()  # Model of choice
model.fit(X,Y)              # Train the model

# Make a prediction with model for Cyprus
cyprus_gdp = [[37655.2]]
print(model.predict(cyprus_gdp))

