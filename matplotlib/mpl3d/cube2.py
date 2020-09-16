#!/usr/bin/env python
"""

"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


oldway = False
#oldway = True

if oldway:
    x = [0, 200, 100, 100]
    y = [0,   0, 100,   0]
    z = [0,   0,   0, 100]
    indices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]   
    tupleList = zip(x, y, z)
    faces = np.array([[tupleList[indices[ix][iy]] for iy in range(len(indices[0]))] for ix in range(len(indices))])

else:
    verts = np.array( [
        [0,0,0],
        [200,0,0],
        [100,100,0],
        [100,0,100]
    ], dtype=np.float32)

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    ## each integer corresponds to a vertex triplet in the above array
    ## each group of three integers corresponds to a triangle of vertices  
    indices = np.array([  
        [0, 1, 2], 
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]], dtype=np.int32) 

    assert indices.max() < len(verts) 
    faces = verts[indices]
pass


poly3d = Poly3DCollection(faces, linewidths=1, edgecolor='k')   # facecolors='w'
poly3d.set_facecolor((0,0,1,0.1))

ax.scatter(x,y,z)
ax.add_collection3d(poly3d) 

plt.show()
