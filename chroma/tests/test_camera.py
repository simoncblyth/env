#!/usr/bin/env python
"""

This is SEG

"""


import chroma
from chroma.loader import load_geometry_from_string
from chroma.camera import Camera


import os
p = os.path.expandvars("$VIRTUAL_ENV/src/chroma/chroma/models/liberty.stl")
geo = load_geometry_from_string(p)
print geo
print geo.mesh 
print geo.bvh

camera_kwargs = {}
size = (400, 400 )

cam = Camera(geo, size, **camera_kwargs)
print "start"
cam.start()
print "join"
cam.join()





