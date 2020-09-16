#!/usr/bin/env python
"""
https://stackoverflow.com/questions/44881885/python-draw-parallelepiped

"""
from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

import numpy as np

plt.ion()

fig = plt.figure()
ax = plt.axes(projection="3d")

plt.show()
