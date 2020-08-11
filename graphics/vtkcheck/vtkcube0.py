
# https://docs.pyvista.org/examples/00-load/create-poly.html#sphx-glr-examples-00-load-create-poly-py

import numpy as np
import pyvista as pv

# mesh points
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0.5, 0.5, -1]])

# mesh faces
faces = np.hstack([[4, 0, 1, 2, 3],  # square
                   [3, 0, 1, 4],     # triangle
                   [3, 1, 2, 4]])    # triangle

surf = pv.PolyData(vertices, faces)

# plot each face with a different color
surf.plot(scalars=np.arange(3), cpos=[-1, 1, 0.5])
