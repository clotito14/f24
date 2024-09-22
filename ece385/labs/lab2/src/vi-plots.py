import matplotlib.pyplot as plt
import numpy as np

# Example arrays
v = np.array([30, 60, 90, 115, 120, 125])
i = np.array([0.085, 0.18, 0.29, 0.435, 0.49, 0.5])

# Plot the data
plt.plot(i, v, color='red')

# Add labels and a title
plt.xlabel('Excitation Current (A)')
plt.ylabel('Excitation Voltage (V)')
plt.title('Excitation Voltage v. Excitation Current')

# Show the plot
plt.grid(True)
plt.show()