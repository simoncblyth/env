#!/usr/bin/env python
"""
mandelbrot.py 
==============


"""
import numpy as np
SIZE = np.array([1280, 720])
import matplotlib.pyplot as plt

def read_npy(path):
    path = os.path.expandvars("$FOLD/a.npy")
    c = np.load(path)
    d = dict()
    for line in open(path.replace(".npy","_meta.txt")).read().splitlines():
        key, val = line.split(":")
        d[key] = float(val)
    pass 
    return c, d 

if __name__ == '__main__':

    c, d = read_npy("$FOLD/a.npy")

    print( "c.min() %s c.max() %s " % (c.min(), c.max()))
    scale = float(os.environ.get("SCALE", 1))

    extent = (d["xmin"], d["xmax"], d["ymin"], d["ymax"] )
    cmap = plt.cm.prism
    #cmap = None

    fig, ax = plt.subplots(figsize=SIZE/100.)  
    ax.imshow(c*scale, extent=extent, cmap=cmap) 
    fig.show()
