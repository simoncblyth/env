#!/usr/bin/env python
"""


In [7]: bottom_right_(0.5)                                                                                                               
Out[7]: array([640., 360., 640., 360.], dtype=float32)

In [8]: bottom_right_(0.6)                                                                                                               
Out[8]: array([768., 432., 512., 288.], dtype=float32)

In [9]: 1152+128                                                                                                                         
Out[9]: 1280

In [10]: bottom_right_(0.7)                                                                                                              
Out[10]: array([896., 504., 384., 216.], dtype=float32)

INFO:env.doc.downsize:downsize GPU_CPU_geometry_model.png to create GPU_CPU_geometry_model_half.png 2188px_1096px -> 1094px_548px

"""
import numpy as np

full = np.array([1280,720], dtype=np.float32 )

other = np.array([1094,548], dtype=np.float32 )


bottom_right_ = lambda f:np.array([ full[0]*f, full[1]*f, full[0]*(1.-f), full[1]*(1.-f) ], dtype=np.float32 )  

other_bottom_right_ = lambda f:np.array([ other[0]*f, other[1]*f, full[0] - other[0]*f, full[1] - other[1]*f ], dtype=np.float32 )

fmt_ = lambda a:"%dpx_%dpx %dpx_%dpx" % tuple(a.astype(np.int32))  


if __name__ == '__main__':

   print("full:%s (standard size of slide)" % full )
   for f in [1.0, 0.5, 0.6, 0.7]:
       fbr = bottom_right_(f)
       print("bottom_right_(%3.1f)" % f, fbr, fmt_(fbr))
   pass   
   print("other:%s (original or aspect preserving scaled size of img to place at bottom right) " % other )
   for f in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
       obr = other_bottom_right_(f)
       print("other_bottom_right_(%3.1f)" % f,  obr, fmt_(obr))
   pass   


         
