#!/usr/bin/evn python

"""
  http://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()

plt.axis('equal')
ax.set_xlim(-10.,10.)
ax.set_ylim(-10.,10.)


patches = []

bb = mtransforms.Bbox([[-2,-2],[2,2]])
bb2 = mtransforms.Bbox([[0,0],[0,0]])

kwa = {}

kwa['width'] = 0.1
kwa['width'] = None
#kwa['clip_box'] = bb
kwa['clip_on'] = True

rect = mpatches.Rectangle((-1,-1),2,2, facecolor="r", edgecolor="none")
ax.add_artist(rect)

w1 = mpatches.Wedge([0.,0.], 5.0,  -30., 30., **kwa)
w2 = mpatches.Wedge([0.,0.], 5.0,  90., 110., **kwa)
w3 = mpatches.Wedge([0.,0.], 5.0,  -110., -90., **kwa)

w1.set_clip_path(rect)

ax.add_artist(w1)
ax.add_artist(w2)
ax.add_artist(w3)



plt.show()
