#!/usr/bin/env python

import pyvista as pv

a = 0.9
def spider_cage(x, y, z):
    x2 = x * x
    y2 = y * y
    x2_y2 = x2 + y2
    return (
        np.sqrt((x2 - y2)**2 / x2_y2 + 3 * (z * np.sin(a))**2) - 3)**2 + 6 * (
        np.sqrt((x * y)**2 / x2_y2 + (z * np.cos(a))**2) - 1.5
    )**2


# create a uniform grid to sample the function with
n = 100
x_min, y_min, z_min = -5, -5, -3

dims=(n, n, n)
spacing=(abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2)
origin=(x_min, y_min, z_min)

grid = pv.UniformGrid(dims, spacing, origin)

x, y, z = grid.points.T

grid.point_arrays["values"] = spider_cage(x, y, z)


mesh = grid.contour(1, "values", method='marching_cubes', rng=[1, 0])
dist = np.linalg.norm(mesh.points, axis=1)
mesh.plot(
    scalars=dist, smooth_shading=True, specular=5,
    cmap="plasma", show_scalar_bar=False
)

