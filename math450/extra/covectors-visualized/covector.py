"""
Hoping to try and visualize covectors
"""

import numpy as np
import matplotlib.pyplot as plt


# define a function f on R2
def f(x) -> float :
    return x


# define function to find plot 1-forms on f in R2
def show_one_forms_r2(f, interval : tuple, n=100) : 
    # unpack interval on x-axis
    a, b = interval
    x = np.linspace(a,b,n)
    
    # plot f on interval
    plt.plot(x, f(x), color='green')
    plt.xlabel('$x$')
    plt.ylabel('$f$')
    plt.title(f'$f(x)$ from ({a}, {b})')
    plt.show()
    plt.close()