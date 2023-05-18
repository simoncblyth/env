#!/usr/bin/env python
"""
https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html


We can rewrite the line equation as y = Ap, 
where A = [[x 1]] and p = [[m], [c]].

"""
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])


A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]


import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Original data', markersize=10)

plt.plot(x, m*x + c, 'r', label='Fitted line')

plt.legend()

plt.show()



