import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data
t = np.linspace(0, 1, 100)
x1 = 2 - t
y1 = t
z1 = -2 
x2 = 2 / (1 + t)
y2 = t**2
z2 = -2
# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D line
ax.plot(x1, y1, z1, c='b')
ax.plot(x2, y2, z2, c='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_box_aspect([1,1,1])

# Show the plot
plt.show()