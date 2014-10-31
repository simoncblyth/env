#!/usr/bin/env python
"""
transform_cache.py 
====================

The transform cache is a dict of global-to-local homogenous 4x4 
transform matrices keyed on PmtId. The transforms are serialized directly 
from the G4AffineTransforms using G4DAETransformCache 

* the unfamiliar row-vector convention used in implementation of G4AffineTransform
  is retained  

* NB avoid copy/pasting numbers dumped from MockNuWa
  as there is insufficient precision retained in the string formatting
  due to the large values of some coordinates, **this can be misleading** 

 
::

    (chroma_env)delta:env blyth$ transform_cache.sh 0x1010101 -i
    ['0x1010101']
    0x1010101 16843009 
    [[      0.           0.7615      -0.6481       0.    ]
     [     -0.           0.6481       0.7615       0.    ]
     [      1.           0.           0.           0.    ]
     [   8842.5     532069.326   599608.6129       1.    ]]
    Python 2.7.8 (default, Jul 13 2014, 17:11:32) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: g2l = tc[0x1010101]

    In [2]: g2l   # global to local in row-vector convention
    Out[2]: 
    array([[      0.    ,       0.7615,      -0.6481,       0.    ],
           [     -0.    ,       0.6481,       0.7615,       0.    ],
           [      1.    ,       0.    ,       0.    ,       0.    ],
           [   8842.5   ,  532069.326 ,  599608.6129,       1.    ]])

    In [3]: l2g = invert_homogenous(g2l.T).T   # as invert_homogenous assumes the other convention

    In [4]: l2g    # local to global in row-vector convention, obtained by inverting g2l 
    Out[4]: 
    array([[      0.    ,      -0.    ,       1.    ,       0.    ],
           [      0.7615,       0.6481,       0.    ,       0.    ],
           [     -0.6481,       0.7615,       0.    ,       0.    ],
           [ -16572.8991, -801469.6472,   -8842.5   ,       1.    ]])

    In [5]: l = np.array([0,0,0,1])   # local origin

    In [6]: g = np.dot( l, l2g )      # pre-mult, row vector convention

    In [7]: g     
    Out[7]: array([ -16572.8991, -801469.6472,   -8842.5   ,       1.    ])

    In [8]: np.dot( g, g2l )          
    Out[8]: array([ 0., -0., -0.,  1.])

    In [9]: np.dot( g2l.T, g  )  # post-mult with transform to allow more familial column vector convention
    Out[9]: array([ 0., -0., -0.,  1.])

    In [12]: np.allclose( np.dot( g2l, l2g ), np.identity(4) )
    Out[12]: True

    In [13]: np.allclose( np.dot( l2g, g2l ), np.identity(4) )
    Out[13]: True


"""

import logging, sys
log = logging.getLogger(__name__)


import IPython
import os
import numpy as np

from env.geant4.geometry.collada.g4daeview.daeutil import invert_homogenous



class TransformCache(dict):
    """
    4x4 homogenous matrices corresponding to the G4AffineTransformation 
    objects of all SD (PMTs) persisted to the transform cache  

    Keys are currently the volume index for debugging, 
    but intended to use PmtId once stabilized.
    """
    def __init__(self, archivedir=None): 
        if archivedir is None: 
            archivedir = os.environ['G4DAECHROMA_CACHE_DIR'] # define with: export-;export-export 
        pass
        data = np.load(archivedir + "/data.npy")
        key  = np.load(archivedir + "/key.npy")
        assert len(key) == len(data) 
        dict.__init__(self,zip(key,data))

    def dump(self, *keys):
        keys = map(lambda _:int(_,16), keys )
        if len(keys) > 0: 
            filter_ = lambda k:k in keys 
        else: 
            filter_ = lambda k:k 

        for k in filter(filter_,sorted(self)):
            print "0x%x %d " % (k, k)
            print self[k]






def parse_args(doc):
    import argparse
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-l","--loglevel", default="INFO", help="INFO/DEBUG/WARN/..")  
    parser.add_argument("-i","--ipython", action="store_true", help="Enter embedded IPython shell")  
    parser.add_argument("keys", nargs='+', help="PMTID in hex")  
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    return args



def main():
    args = parse_args(__doc__)
    np.set_printoptions(suppress=True, precision=4)
    keys = args.keys 
    print keys  

    tc = TransformCache()
    tc.dump(*keys)

    if args.ipython:
        IPython.embed()

     
if __name__ == '__main__':
     main()




