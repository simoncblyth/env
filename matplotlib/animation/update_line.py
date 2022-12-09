"""
https://matplotlib.org/stable/gallery/animation/random_walk.html

::

    In [1]: line, = ax.plot([], [], lw=2)

    In [2]: line
    Out[2]: <matplotlib.lines.Line2D at 0x169878190>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)



points = np.array( [[0,0], [1,1]], dtype=np.float32 )


def update(num):
    print(num)
    line = lines[0]

    p = points.copy()     
    p[-1] = [num/40, num/40]
    line.set_data(p.T)

    return lines


num_steps = 40

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim(0,1)
ax.set_ylim(0,1)

lines = [ax.plot([], [], [])[0],]
line = lines[0]    ## lines.Line2D

ani = animation.FuncAnimation(fig, update, num_steps+1, interval=200, repeat=False)

plt.show()




