#!/usr/bin/env python
"""
check_volume_transforms
===========================

Comparing 

* homogenous 4x4 matrix constructed by pycollada **objects** 
  binding of the unbound matrices from the COLLADA export 

* Geant4 transforms written by idmap, obtained 
  from touchable histories created in an artifical 
  full traverse, see *idmap.py*

Observations
--------------

* had to loosen tolerance to succeed with allclose

Definitions matching
---------------------

Need reminder on homogenous 4x4 to understand why need to do 
the below to get a match, usual problem is order of rotation 
and translation : need to read definitions carefully 
to check if there is an mis-interpretation somewhere  

::

       g4m = np.identity(4)
       g4m[:3,:3] = node.g4rot 
       g4m[:3,3] =  np.dot( node.g4rot, -node.g4tra )  
                     # by inspection this is close for all volumes


Sources of imprecision
-----------------------

The hierarchy of matrices is written into g4dae 
collada docs parsed by pycollada and product matrix is formed at 
pycollada binding.

The geant4 matrix is obtained from touchable history 
top transform.

IPython
------------

::

    In [14]: np.set_printoptions(precision=5, suppress=True)

    In [15]: n1000.boundgeom.matrix
    Out[15]: 
    array([[     -0.53929,      -0.84111,      -0.04126,  -19953.46094],
           [      0.84212,      -0.53864,      -0.02643, -799555.     ],
           [      0.     ,      -0.049  ,       0.9988 ,   -1369.56641],
           [      0.     ,       0.     ,       0.     ,       1.     ]], dtype=float32)

    In [16]: n1000.g4rot
    Out[16]: 
    array([[-0.53929, -0.84111, -0.04126],
           [ 0.84212, -0.53864, -0.02643],
           [ 0.     , -0.049  ,  0.9988 ]])

    In [17]: n1000.g4tra
    Out[17]: array([ 662561. , -447524. ,  -20583.8])

    In [18]: np.dot( n1000.g4rot , n1000.g4tra )
    Out[18]: array([  19953.31029,  799555.02276,    1369.59714])

    In [19]: np.dot( n1000.g4rot , -n1000.g4tra )
    Out[19]: array([ -19953.31029, -799555.02276,   -1369.59714])



"""

import os, sys, logging
import numpy as np
import IPython as IP
from g4daenode import DAENode
log = logging.getLogger(__name__)

if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO) 
   np.set_printoptions(precision=5, suppress=True)
   DAENode.init() 
   for i, node in enumerate(DAENode.registry):
       matrix = node.boundgeom.matrix   # homogenous 4x4 matrix constructed by pycollada binding from the COLLADA export 

       g4m = np.identity(4)
       g4m[:3,:3] = node.g4rot 
       g4m[:3,3] =  np.dot( node.g4rot, -node.g4tra )  # by inspection this is close 

       # had to tweak tolerance to succeed with allclose
       if not np.allclose( matrix, g4m, rtol=3.e-4 ):
           print i, node
           print matrix
           print g4m 









   








