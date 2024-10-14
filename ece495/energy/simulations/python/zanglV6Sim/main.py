"""
Filename: main.py
Description: This script averages the CSV power data from Zangl V6 LTSPICE sim made by Aaron

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
filePath = ".\\zangl_output_power.csv"

# read in CSV data
data = pd.read_csv(filePath, delimiter="\t")
data_avg = data['V(out)*I(RL)'].values.mean()
print('Zangl V6 Circuit Simulation Outputs:')
print('------------------------------------')
print(f"Average Power: {data_avg*1000:.4f} mW\n")

# convert power to mW
data['V(out)*I(RL)'] = data['V(out)*I(RL)'] * 1000 

# plot
#data.plot(x='time', y='V(VLINE,N001)*I(Ceh)', kind='line')
data.plot(x='time', y='V(out)*I(RL)', kind='line', c='g')
plt.xlabel('Time (s)')
plt.ylabel('$P_{\text{out}}$ (mW)')
plt.title('EFEH Output Power - Zangl V6 Simulation')
plt.xlim(0,60)
plt.ylim(0,15)
plt.show()