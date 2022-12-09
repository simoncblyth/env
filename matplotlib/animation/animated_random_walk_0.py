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

def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    #start_pos = np.random.random(3)
    start_pos = np.array( [0.5,0.5,0.5] )
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))  # (40, 3)
    walk = start_pos + np.cumsum(steps, axis=0)  # (40, 3)
    return walk


def update_lines(num, walks, lines):
    print(num)
    for line, walk in zip(lines, walks):
        line.set_data(walk[:num, :2].T)
    pass
    return lines



num_steps = 40
walks = [random_walk(num_steps),] 
walk = walks[0]

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.scatter( walk[:,0], walk[:,1] )

lines = [ax.plot([], [], [])[0],]
line = lines[0]    ## lines.Line2D

ani = animation.FuncAnimation(fig, update_lines, num_steps+1, fargs=(walks, lines), interval=200, repeat=False)

plt.show()




