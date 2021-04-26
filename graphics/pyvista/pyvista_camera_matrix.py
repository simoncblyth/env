#!/usr/bin/env python
"""
https://github.com/pyvista/pyvista-support/issues/85

"""
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import vtk

def trans_to_matrix(trans):
    """Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    matrix = vtk.vtkMatrix4x4()
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            matrix.SetElement(i, j, trans[i, j])
    return matrix

mesh = pv.Sphere()


off_screen = False
reset_camera = True

plotter = pv.Plotter(off_screen=off_screen)
plotter.add_mesh(mesh, reset_camera=reset_camera)

modelTransform = np.array([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
], dtype=np.float32)

f = 1
projTransform = np.array([
  [f, 0, 0, 0],
  [0, f, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
], dtype=np.float32)

plotter.camera.SetModelTransformMatrix(trans_to_matrix(modelTransform))
plotter.camera.SetExplicitProjectionTransformMatrix(trans_to_matrix(projTransform))
plotter.camera.SetUseExplicitProjectionTransformMatrix(1)

#plotter.camera_set = True   # this avoids the default camera position from pyvista.rcParams being used

if off_screen:
    cam, img = plotter.show(return_img=True)   # not working in my version
    #img = plotter.image
    plt.imshow(img)
    plt.show()
else:
    cam  = plotter.show()
pass



