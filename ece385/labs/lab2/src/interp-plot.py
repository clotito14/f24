import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Example data in Q1
x_q1 = np.array([1, 2, 3, 4, 5])
y_q1 = np.array([10, 20, 15, 25, 30])

# Reflect data into Q3 by negating both x and y values
x_q3 = -x_q1
y_q3 = -y_q1

# Interpolate within Q1 for smoother curves
f = interp1d(x_q1, y_q1, kind='linear')  # Linear interpolation function
x_new_q1 = np.linspace(min(x_q1), max(x_q1), 50)  # Interpolated x values in Q1
y_new_q1 = f(x_new_q1)  # Interpolated y values in Q1

# Reflect interpolated points into Q3
x_new_q3 = -x_new_q1
y_new_q3 = -y_new_q1

# Combine Q1 and Q3 data points to create a connected curve
x_combined = np.concatenate([x_new_q3[::-1], x_new_q1])  # Concatenate Q3 and Q1
y_combined = np.concatenate([y_new_q3[::-1], y_new_q1])  # Concatenate Q3 and Q1

# Create the plot
fig, ax = plt.subplots()

# Plot the connected curve
ax.plot(x_combined, y_combined, color='orange', label='Connected Q1 and Q3 Data')

# Move the x-axis and y-axis to the center
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide the top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Add arrows to the axes
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Set ticks for all four quadrants
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Add labels, title, and grid
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Graph in 4 Quadrants')
plt.grid(True)

# Show the plot
plt.show()
