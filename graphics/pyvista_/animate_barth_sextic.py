#!/usr/bin/env python
"""
https://docs.pyvista.org/examples/01-filter/flying_edges.html?highlight=marching%20cubes

"""
import numpy as np
import pyvista as pv

phi = (1 + np.sqrt(5)) / 2
phi2 = phi * phi

def barth_sextic(x, y, z):
    x2 = x * x
    y2 = y * y
    z2 = z * z
    arr = (
        3 * (phi2 * x2 - y2) * (phi2 * y2 - z2) * (phi2 * z2 - x2)
        - (1 + 2 * phi) * (x2 + y2 + z2 - 1)**2
    )
    nan_mask = x2 + y2 + z2 > 3.1
    #arr[nan_mask] = np.nan
    return arr

# create a uniform grid to sample the function with
n = 100
k = 2.0
x_min, y_min, z_min = -k, -k, -k

dims=(n, n, n)
spacing=(abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2)
origin=(x_min, y_min, z_min)

grid = pv.UniformGrid(dims, spacing, origin)
x, y, z = grid.points.T

# sample and plot

grid.point_arrays["values"] = barth_sextic(x, y, z)


if 1:
    mesh = grid.contour(1, "values", method='marching_cubes', rng=[-0.0, 0])
    dist = np.linalg.norm(mesh.points, axis=1)
    mesh.plot(
        scalars=dist, smooth_shading=True, specular=5,
        cmap="plasma", show_scalar_bar=False
    )




angle_to_range = lambda angle:-2*np.sin(angle)

mesh = grid.contour(1, "values", method='marching_cubes', rng=[angle_to_range(0), 0])


dist = np.linalg.norm(mesh.points, axis=1)

pl = pv.Plotter()
pl.add_mesh(mesh, scalars=dist, smooth_shading=True, specular=5, rng=[0.5, 1.5], cmap="plasma", show_scalar_bar=False)
pl.open_gif('barth_sextic.gif')

for angle in np.linspace(0, np.pi, 15)[:-1]:
    new_mesh = grid.contour(1, "values", method='marching_cubes', rng=[angle_to_range(angle), 0])
    mesh.overwrite(new_mesh)
    pl.update_scalars(np.linalg.norm(new_mesh.points, axis=1), render=False)
    pl.write_frame()
pass

pl.show()



