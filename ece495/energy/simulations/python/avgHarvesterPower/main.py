"""
Filename: main.py
Description: This script averages the CSV power data from Primary Filter Capacitor Voltage LTSPICE sim

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
filePath = ".\\cap_power.csv"

# read in CSV data
data = pd.read_csv(filePath, delimiter="\t")
data_avg = data['V(VLINE,N001)*I(Ceh)'].values.mean()
print(f"Average Power: {data_avg} W")

# convert power to mW
data['V(VLINE,N001)*I(Ceh)'] = data['V(VLINE,N001)*I(Ceh)'] * 1000 

# plot
#data.plot(x='time', y='V(VLINE,N001)*I(Ceh)', kind='line')
data.plot(x='time', y='V(VLINE,N001)*I(Ceh)', kind='line', c='g')
plt.xlabel('Time (s)')
plt.ylabel('$P_{C_{EH}}$ (mW)')
plt.title('Instantaneous Power through Energy Harvester for $C_{EH} = 1$pF')
plt.xlim(30,30.06)
plt.show()