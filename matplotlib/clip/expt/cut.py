#!/usr/bin/env python
"""
https://stackoverflow.com/questions/42491595/cut-parts-of-plotted-artists

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Circle, Ellipse


fig, ax = plt.subplots(figsize=(6,6))



def make_rect(xc, yc, hx, hy):
    """
    :param xc: 
    :param yc: 
    """
    x0 = xc - hx
    y0 = yc - hy
    rect = Rectangle((x0, y0), hx*2, hy*2)
    return rect

def create_clip_rect(rect, **kwa):
    rec_path = rect.get_path()  
    rec_path = rect.get_patch_transform().transform_path(rec_path) 
    path = Path(vertices=rec_path.vertices, codes=rec_path.codes ) 
    patch = PathPatch(path, **kwa)
    return patch


ex = 100
ey = 100 
e = Ellipse([0,0], width=ex*2, height=ey*2, fc='w', ec='b')
ax.add_patch(e)

e2 = Ellipse([0,ey/2+20], width=ex, height=ey, fc='w', ec='b' )
ax.add_patch(e2)

rect = make_rect(0,50,100,50)
clip = create_clip_rect(rect, facecolor='w', edgecolor='w')
ax.add_patch(clip)


ax.set_xlim(-110,110)
ax.set_ylim(-110,110)

plt.tight_layout()
plt.show()





