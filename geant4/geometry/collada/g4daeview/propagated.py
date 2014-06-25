#!/usr/bin/env python
"""

From ipython::

    In [132]: cd /Users/blyth/env/geant4/geometry/collada/g4daeview
    /Users/blyth/env/geant4/geometry/collada/g4daeview

    In [133]: run propagated.py
    [[  -1    0    2    2]




::

    In [246]: a['last_hit_triangle'][::10]
    Out[246]: 
    array([[  -1,    0,    2,    2],
           [  -1,    1,    1,    1],
           [  -1,    2,    1,    1],
           ..., 
           [  -1, 4162,    1,    1],
           [  -1, 4163,   15,   15],
           [  -1, 4164,  100,  100]], dtype=int32)

    In [247]: a['last_hit_triangle'][::10][:,3]
    Out[247]: array([  2,   1,   1, ...,   1,  15, 100], dtype=int32)


    In [248]: np.where( a['last_hit_triangle'][::10][:,3] == 5 )
    Out[248]: 
    (array([  63,   66,  108,  136,  144,  160,  171,  172,  201,  250,  398,
            413,  429,  526,  772,  894,  899,  946, 1052, 1083, 1256, 1285,
           1441, 1625, 1650, 1755, 2003, 2009, 2027, 2200, 2259, 2276, 2374,
           2507, 2509, 2544, 2554, 2654, 2697, 2875, 2879, 2884, 2911, 2953,
           3066, 3072, 3090, 3095, 3141, 3223, 3224, 3266, 3348, 3397, 3423,
           3462, 3473, 3527, 3614, 3650, 3705, 3734, 3831, 3867, 3914, 4044,
           4096, 4102, 4145]),)

      ## when not truncated should add one to the slots to get the count 

    In [261]: a['last_hit_triangle'][630:640,:3]
    Out[261]: 
    array([[ -1,  63,   5],
           [576,   0,   0],
           [288,   0,   0],
           [577,   0,   0],
           [580,   0,   0],
           [ -1,   0,   0],
           [  0,   0,   0],
           [  0,   0,   0],
           [  0,   0,   0],
           [  0,   0,   0]], dtype=int32)


    In [69]: a['flags'][::10,0]
    Out[69]: array([  2,   2,   2, ..., 514,   2, 610], dtype=uint32)




"""
import os
import numpy as np

from chroma.event import mask2arg_, count_unique



path = os.path.expanduser("~/e/propagated.npz")
#path = "propagated.npz"

with np.load(path) as npz:
     a = npz['propagated']

print a['last_hit_triangle'][::10]


u = count_unique( a['flags'][::10,0] )
print "\n".join(map(lambda _:"%5s %-80s %s " % ( _[0], mask2arg_(_[0]), _[1] ),u) )


