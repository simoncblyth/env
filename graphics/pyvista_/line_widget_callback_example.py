#!/usr/bin/env python
"""
https://docs.pyvista.org/examples/03-widgets/line-widget.html

"""

import numpy as np

import pyvista as pv
from pyvista import examples

pv.set_plot_theme('document')

mesh = examples.download_kitchen()
furniture = examples.download_kitchen(split=True)

arr = np.linalg.norm(mesh['velocity'], axis=1)
clim = [arr.min(), arr.max()]


p = pv.Plotter()
p.add_mesh(furniture, name='furniture', color=True)
p.add_mesh(mesh.outline(), color='black')
p.add_axes()

def simulate(pointa, pointb):
    streamlines = mesh.streamlines(n_points=10, max_steps=100,
                                   pointa=pointa, pointb=pointb,
                                   integration_direction='forward')
    p.add_mesh(streamlines, name='streamlines', line_width=5,
               render_lines_as_tubes=True, clim=clim)

p.add_line_widget(callback=simulate, use_vertices=True)
p.show()
