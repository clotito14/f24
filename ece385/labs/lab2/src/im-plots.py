import matplotlib.pyplot as plt
import numpy as np

# Example arrays
v = np.array([115, 120, 125])
i_temp = []

for x in v:
    val = x / 265.8
    i_temp.append(val)

i = np.array(i_temp)

# Plot the data
plt.scatter(v, i, color='blue')

# Add labels and a title
plt.ylabel('Magnetizing Current (A)')
plt.xlabel('Excitation Voltage (V)')
plt.title('Excitation Voltage v. Magnetizing Current')

# Show the plot
plt.grid(True)
plt.show()