#!/usr/bin/env python
"""
https://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work

https://stackoverflow.com/questions/10192011/clipping-in-matplotlib-why-doesnt-this-work

set_clip_box
set_clip_path

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.ion()


def make_rect(cx,cy,hx,hy, **kwa):
    x = cx-hx
    y = cy-hy 
    r = Rectangle((x,y),2*hx,2*hy, **kwa)
    return r


ax = plt.axes()

for ix in range(10):
    for iy in range(10):

        x = float(ix-5)/5.
        y = float(iy-5)/5.

        r = make_rect(x, y+0.05, 0.1, 0.05, fc='w', ec='b')

        c = Circle((x,y),0.1, fc='w', ec='b')
        ax.add_artist(c)
        ax.add_artist(r)

        if ix == 5 and iy == 5:
            c.set_clip_path(r)
        pass
    pass
pass


plt.axis('equal')
plt.axis((-2,2,-2,2))

plt.show()



