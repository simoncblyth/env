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

    def dump(self):
        for k in sorted(self):
            print k, self[k]



if __name__ == '__main__':
     tc = TransformCache()
     #tc.dump()

     




