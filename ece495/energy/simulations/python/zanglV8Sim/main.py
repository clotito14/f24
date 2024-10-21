"""
Filename: main.py
Description: This script averages the CSV power data Zangl V8, then plots against primary coil inductance. 

Author: Chase Lotito
Email: chase.lotito@siu.edu
Created: 2024-10-12
Modified: n/a
Version: 1.0.1

Dependencies:
    - numpy >= 1.19.0
    - pandas >= 1.1.0
    - matplotlib >= 3.3.0

Usage:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# constants
filePath = ".\\step.csv"
power_node = 'V(vdc,COM)*I(RL)'
time = 'time'

# lists to be filled
avg_power = []
inductances = [100, 1000, 10000, 30000, 50000, 100000]

for i in range(1, 7):
    filePath = f'.\\step{i}.csv'
    
    # read in csv data
    data = pd.read_csv(filePath, delimiter='\t')
    data_avg = data[power_node].values.mean()
    avg_power.append(data_avg*1000)          # add in terms of mW

max_avg_power = max(avg_power)
print(f'Max Average Power: {max_avg_power:.3f}mW')

# plot
plt.plot(inductances, avg_power, c='g')
plt.xlabel('Primary Coil Inductance $L_p$ ($H$)')
plt.ylabel('Output Power $P_{out}$ ($mW$)')
plt.title('Output Power v. Transformer Inductance')
plt.show()