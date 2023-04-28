#!/usr/bin/env python
"""
https://matplotlib.org/2.0.2/examples/pylab_examples/quiver_demo.html

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

x = np.arange(0, 2 * np.pi, .2)
y = np.arange(0, 2 * np.pi, .2)

u = np.cos(x)
v = np.sin(y)


X, Y = np.meshgrid(x, y)     
U = np.cos(X)
V = np.sin(Y)
M = np.hypot(U, V)  # Given the "legs" of a right triangle, return its hypotenuse.

for sym in "x y X Y U V M".split():
    expr = "%s.shape" % sym
    print("%s  : %s " % (expr, str(eval(expr))))
pass


plt.figure()
plt.title('Arrows scale with plot width, not view')
#Q = plt.quiver(X, Y, U, V, units='width')
Q = plt.quiver(x, y, u, v, units='width')
plt.scatter( x, y, s=5, c="r" )


"""
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')


plt.figure()
plt.title("pivot='mid'; every third arrow; units='inches'")
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], pivot='mid', units='inches')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)

plt.figure()
plt.title("pivot='tip'; scales with x view")
Q = plt.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022, scale=1 / 0.15)
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
plt.scatter(X, Y, color='k', s=5)
"""

plt.show()


