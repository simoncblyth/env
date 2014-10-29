#!/usr/bin/env python
"""

::

    delta:collada blyth$ ipython.sh transform_cache.py 

    In [2]: np.set_printoptions(suppress=True, precision=4)

    In [3]: tc[3200]
    Out[3]: 
    array([[      0.    ,       0.7615,      -0.6481,    8842.5   ],
           [      0.    ,       0.6481,       0.7615,  532069.326 ],
           [      1.    ,      -0.    ,       0.    ,  599608.6129],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [5]: len(tc)
    Out[5]: 684



::

    delta:collada blyth$ ipython.sh transform_cache.py /tmp/DybG4DAEGeometry.cache 0x1010101 0x1020701 
    ...
    0x1010101 16843009 
    [[      0.           0.7615      -0.6481    8842.5   ]
     [     -0.           0.6481       0.7615  532069.326 ]
     [      1.           0.           0.      599608.6129]
     [      0.           0.           0.           1.    ]]
    0x1020701 16910081 
    [[      0.           0.7615      -0.6481    5842.5   ]
     [     -0.           0.6481       0.7615  532818.8074]
     [      1.           0.           0.      605301.4893]
     [      0.           0.           0.           1.    ]]

    In [1]: 




"""
import os
import numpy as np


class TransformCache(dict):
    """
    4x4 homogenous matrices corresponding to the G4AffineTransformation 
    objects of all SD (PMTs) persisted to the transform cache  

    Keys are currently the volume index for debugging, 
    but intended to use PmtId once stabilized.
    """
    def __init__(self, archivedir=None): 
        if archivedir is None: 
            archivedir = os.environ['DAE_NAME_DYB_TRANSFORMCACHE'] # define with: export-;export-export 
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



if __name__ == '__main__':
     np.set_printoptions(suppress=True, precision=4)
     import sys

     archivedir = None
     if len(sys.argv) > 1:
         archivedir = sys.argv[1]

     keys = []     
     if len(sys.argv) > 2:
         keys = sys.argv[2:]

     tc = TransformCache(archivedir)
     tc.dump(*keys)

     




