"""

https://docs.pyvista.org/examples/02-plot/gif.html

To view the gif animation open the folder, select the gif and press spacebar (is that quickview?)

Animation also pays via web browser when using localhost urls::

    open file://$PWD/wave.gif   ## shows individual frames, not the movie
    cp wave.gif /env/presentation/wave.gif
    open http://localhost/env/presentation/wave.gif


"""

import numpy as np
import pyvista as pv


x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x ** 2 + y ** 2)
z = np.sin(r)

# Create and structured surface
grid = pv.StructuredGrid(x, y, z)

# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_mesh(grid, scalars=z.ravel(), smooth_shading=True)

# Open a gif
plotter.open_gif("wave.gif")

pts = grid.points.copy()

# Update Z and write a frame for each updated position
nframe = 15
for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    z = np.sin(r + phase)
    pts[:, -1] = z.ravel()
    plotter.update_coordinates(pts, render=False)
    plotter.update_scalars(z.ravel(), render=False)

    # must update normals when smooth shading is enabled
    plotter.mesh.compute_normals(cell_normals=False, inplace=True)
    plotter.render()
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()

