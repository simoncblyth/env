#!/usr/bin/env python

import matplotlib as mp
import numpy as np
SIZE = np.array([1280, 720] )

def xy_grid_coordinates(nx=11, ny=11, sx=100., sy=100.):
    """
    :param nx:
    :param ny:
    :param sx:
    :param sy:
    :return xyz: (nx*ny,3) array of XY grid coordinates
    """
    x = np.linspace(-sx,sx,nx)
    y = np.linspace(-sy,sy,ny)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros( (nx, ny) )
    xyz = np.dstack( (xx,yy,zz) ).reshape(-1,3)  
    return xyz 


if __name__ == '__main__':

    xyz = xy_grid_coordinates()
    print("xyx: %s " % (str(xyz.shape)))

    fig, ax = mp.pyplot.subplots(figsize=SIZE/100.)
    ax.set_aspect('equal')
    ax.scatter( xyz[:,0], xyz[:,1] )
    fig.show()







