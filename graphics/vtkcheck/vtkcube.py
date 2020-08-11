#!/usr/bin/env python
"""
See cubeplt.py for 3d plotting of the cubes
"""

import numpy as np
from opticks.ana.cube import make_cube
import pyvista as pv

if __name__ == '__main__':

    bbox = np.array([(0,0,0),(100,100,100)], dtype=np.float32)
    points, faces, indices = make_cube(bbox)

    surf = pv.PolyData(points, indices)

    pl = pv.Plotter()
    pl.add_mesh(surf, opacity=0.85, color=True)
    pl.show()

