#!/usr/bin/evn python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()

patches = []


patches.append(mpatches.Wedge([0.,0.], 5.0,  -30., 30., width=0.1))
patches.append(mpatches.Wedge([0.,0.], 5.0,  90., 110., width=0.1))
patches.append(mpatches.Wedge([0.,0.], 5.0,  -110., -90., width=0.1))


for p in patches:
    ax.add_artist(p)

plt.axis('equal')

ax.set_xlim(-10.,10.)
ax.set_ylim(-10.,10.)


plt.show()
