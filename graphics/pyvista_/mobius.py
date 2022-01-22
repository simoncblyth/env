"""
https://docs.pyvista.org/api/core/_autosummary/pyvista.UniformGrid.contour.html


https://github.com/pyvista/pyvista-support/issues/29

970299


In [1]: 99*99*99                                                                                                                                                                                                                           
Out[1]: 970299


"""

import pyvista as pv
a = 0.4
b = 0.1

def f(x, y, z):
    xx = x*x
    yy = y*y
    zz = z*z
    xyz = x*y*z
    xx_yy = xx + yy
    a_xx = a*xx
    b_yy = b*yy
    return (
        (xx_yy + 1) * (a_xx + b_yy)
        + zz * (b * xx + a * yy) - 2 * (a - b) * xyz
        - a * b * xx_yy
    )**2 - 4 * (xx + yy) * (a_xx + b_yy - xyz * (a - b))**2


n = 100
x_min, y_min, z_min = -1.35, -1.7, -0.65

dims=(n, n, n)
spacing=(abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2)
origin=(x_min, y_min, z_min)

grid = pv.UniformGrid(dims, spacing, origin)
x, y, z = grid.points.T

grid.point_arrays["values"] = f(x, y, z)

out = grid.contour(1, scalars="values", rng=[0, 0], method='flying_edges')

out.plot(color='tan', smooth_shading=True)

