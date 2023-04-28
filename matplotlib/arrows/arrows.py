#/usr/bin/env python

import matplotlib.pyplot as plt

x = [0.1,0.2]
y = [0.3,0.4]
dx = [0.5,0.5]
dy = [0.5,0.3]

ax = plt.axes()
ax.arrow(x[0], y[0], dx[0], dy[0], head_width=0.09, head_length=0.1)
ax.arrow(x[1], y[1], dx[1], dy[1], head_width=0.09, head_length=0.1)
plt.show()
