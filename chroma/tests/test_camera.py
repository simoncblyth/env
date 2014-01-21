#!/usr/bin/env python
"""
Avoiding the use of multiprocessing/forking 
for GUI things avoids the pygame SEGV on Mavericks 10.9.1

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
size = (1000, 1000 )

cam = Camera(geo, size, **camera_kwargs)

nofork = True

if nofork:
    cam._run()
else:
    print "start"
    cam.start()
    print "join"
    cam.join()





