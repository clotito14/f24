"""
SIUC - EH2024-20
----------------
Chase Lotito
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
h = 10.668          # height of line (35ft=10.668m)
a = 9e-3            # radius of line, inner radius of C1

# define c1/c2 ratio
def f(x):
    return np.arccosh(h/x) / np.log(x/a)

t1 = np.arange(a, h, 0.01)

# plot c1/c2
plt.figure()
plt.plot(t1, f(t1), 'b', label='$C_1/C_2$')

# Plot a single point at x=5 (you can change this value)
x_single = 30e-2 / 2
y_single = f(x_single)
plt.scatter(x_single, y_single, color='g', marker='x', zorder=5)
plt.text(x_single + 0.2, y_single, f'Zangl, $b=${x_single*100}cm', fontsize=10, color='black')

# plot labels
plt.ylabel('$C_1 / C_2$')
plt.xlabel('Harvester Outer Radius $b$ (m)')
plt.title('$C_1 / C_2$ v. Harvester Radius')
plt.legend()
plt.show()