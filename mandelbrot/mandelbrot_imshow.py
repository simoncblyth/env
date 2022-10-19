#!/usr/bin/env python
"""
"""
from collections import OrderedDict as odict 
import matplotlib.pyplot as plt
import numpy as np


def read_meta(npy_path):
    txt_path = npy_path.replace(".npy","_meta.txt")
    lines = open(txt_path).read().splitlines()
    d = odict()
    for line in lines:
        key, val = line.split(":")
        d[key] = float(val)
    pass 
    return d 


if __name__ == '__main__':

    path = "/tmp/mandelbrot.npy"
    c = np.load(path)
    d = read_meta(path)

    print( "c.min() %s c.max() %s " % (c.min(), c.max()))
    scale = float(os.environ.get("SCALE", 1))

    extent = (d["xmin"], d["xmax"], d["ymin"], d["ymax"] )
    cmap = plt.cm.prism
    #cmap = None

    figsize = np.array([1280, 720])
    fig, ax = plt.subplots(figsize=figsize/100.)  
    ax.imshow(c*scale, extent=extent, cmap=cmap) 
    fig.show()
