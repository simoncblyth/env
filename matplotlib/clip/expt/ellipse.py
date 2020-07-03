#!/usr/bin/env python
"""

https://matplotlib.org/api/path_api.html?highlight=curve4

Path.MOVETO (1)
   1 vertex
   Pick up the pen and move to the given vertex.

Path.CURVE4 (4)
   2 control points, 1 endpoint
   Draw a cubic Bezier curve from the current position, with the given control
   points, to the given end point.

Path.CLOSEPOLY (79)
   1 vertex (ignored)
   Draw a line segment to the start point of the current polyline.
   


vertices
   array-like
   The (N, 2) float array, masked array or sequence of pairs representing the
   vertices of the path.

   If vertices contains masked values, they will be converted to NaNs which are
   then handled correctly by the Agg PathIterator and other consumers of path
   data, such as iter_segments().




"""
import os, sys, argparse, logging
import numpy as np, math 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Circle, Ellipse, PathPatch

log = logging.getLogger(__name__)
sys.path.insert(0, os.path.expanduser("~"))  # assumes $HOME/opticks 


if __name__ == '__main__':

     plt.ion()

     ex = 100
     ey = 100 

     e = Ellipse([0,0], width=ex*2, height=ey*2, fc='w', ec='b')

     fig, ax = plt.subplots(figsize=(6,6))

     ax.set_xlim(-110,110)
     ax.set_ylim(-110,110)

     ax.add_patch(e)

     p = e.get_path() 

     pt = e.get_patch_transform().transform_path(p)

     ax.scatter(pt.vertices[:,0],pt.vertices[:,1], marker="+")



     fig.show()
     

    



