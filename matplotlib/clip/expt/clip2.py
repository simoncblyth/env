#!/usr/bin/env python
"""
https://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Bbox


plt.ion()

# This is in PIXELS
# first tuple : coords of box' bottom-left corner, from the figure's bottom-left corner
# second tuple : coords of box' top-right corner, from the figure's bottom-left corner
clip_box = Bbox(((0,0),(50,50)))


rect = Rectangle((-100,0), 200, 200, ec='none', fc='none')

circle = Circle((0,0),100, fc='none', ec='b')

plt.axis('equal')

fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim(-500,500)
ax.set_ylim(-500,500)

ax.add_artist(rect)
ax.add_artist(circle)


# You have to call this after add_artist()
circle.set_clip_path(rect)

fig.show()



