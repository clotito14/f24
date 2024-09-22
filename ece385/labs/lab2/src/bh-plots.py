import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Calculated magentic flux density and magnetic field 
# intensity from experimental data. 
b = np.array([0.202, 0.404, 0.607, 0.775, 0.809, 0.842])
h = np.array([90.7, 192.1, 309.5, 464.2, 522.9, 533.6])

# Interpoltated into Quadrant 3
b_q3 = -1 * b
h_q3 = -1 * h

# Combine Q1 and Q3 data points to create a connected curve
h_combined = np.concatenate([h_q3[::-1], h])  # Concatenate Q3 and Q1
b_combined = np.concatenate([b_q3[::-1], b])  # Concatenate Q3 and Q1

# Interploate the combined dataset
f = interp1d(h_combined, b_combined, kind='cubic')
h_new = np.linspace(min(h_combined), max(h_combined), 1000)
b_new = f(h_new)

# Plot the data
plt.plot(h_new, b_new, color='purple')

# Add labels and a title
plt.xlabel('$H_{max}$   $(\\frac{\\text{Amp-turns}}{m})$')
plt.ylabel('$B_{max}$   $(T)$')
plt.title('$B_{max}$ v. $H_{max}$ (Interpolated)')

# Show the plot
plt.grid(True)
plt.show()