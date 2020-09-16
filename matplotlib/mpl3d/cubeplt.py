#!/usr/bin/env python
"""
cubeplt.py
============


This is using 3d plotting from matplotlib.



"""
import numpy as np
import matplotlib.pyplot as plt

from opticks.ana.cube import make_cube
from opticks.ana.plt3d import polyplot

if __name__ == '__main__':

    plt.ion()

    bbox = np.array([(0,0,0),(100,100,100)], dtype=np.float32)
    verts, faces, pv_indices = make_cube(bbox)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    polyplot(ax, verts, faces)

    plt.show()


