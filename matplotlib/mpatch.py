import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

plt.axis([-3,3,-3,3])
ax = plt.axes([-3,3,-3,3])
# add a circle
art = mpatches.Circle([0,0], radius = 1, color = 'r', axes = ax)

ax.add_artist(art)

#add another circle
art = mpatches.Circle([0,0], radius = 0.1, color = 'b', axes = ax)

ax.add_artist(art)

print ax.patches

plt.show()
