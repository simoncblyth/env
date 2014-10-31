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

    delta:collada blyth$ ipython.sh transform_cache.py $DAE_NAME_DYB_TRANSFORMCACHE 0x1010101 0x1020701 
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



   M is purportedly the matrix that takes global coords to local in the frame 
   of the PMT  

        
        l = M g 
        l = RT g 

  What g will yield local origin l = (0,0,0,1 ) ?  

       
      M^(-1) l =  g  
        

    In [44]: k = np.identity(4)

    In [45]: k[:3,:3] = m[:3,:3].T

    In [46]: k
    Out[46]: 
    array([[ 0.    , -0.    ,  1.    ,  0.    ],
           [ 0.7615,  0.6481,  0.    ,  0.    ],
           [-0.6481,  0.7615,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]])

    In [47]: k[:,3]
    Out[47]: array([ 0.,  0.,  0.,  1.])

    In [48]: k[:,3] = -m[:,3]

    In [49]: k
    Out[49]: 
    array([[      0.    ,      -0.    ,       1.    ,   -8842.5   ],
           [      0.7615,       0.6481,       0.    , -532069.326 ],
           [     -0.6481,       0.7615,       0.    , -599608.6129],
           [      0.    ,       0.    ,       0.    ,      -1.    ]])

    In [50]: np.dot( k, [0,0,0,1])
    Out[50]: array([  -8842.5   , -532069.326 , -599608.6129,      -1.    ])




    In [85]: g = np.dot( invert_homogenous(m), [0,0,0,1] )

    In [86]: g
    Out[86]: array([-599608.6129, -351578.6214, -399460.1738,       1.    ])

    In [87]: np.dot( m, g )
    Out[87]: array([ 0., -0.,  0.,  1.])





::

    In [90]: mm = np.identity(4)

    In [91]: mm[:3,:3] = m[:3,:3]    ## fill in rotation portion as-is

    In [92]: mm
    Out[92]: 
    array([[ 0.    ,  0.7615, -0.6481,  0.    ],
           [-0.    ,  0.6481,  0.7615,  0.    ],
           [ 1.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  1.    ]])

    In [93]: mm[3]
    Out[93]: array([ 0.,  0.,  0.,  1.])

    In [94]: mm[3] = m[:,3].T     ## fill in row-vector convention translation portion from the mal-placed column-vector convention translation portion  

    In [95]: mm
    Out[95]: 
    array([[      0.    ,       0.7615,      -0.6481,       0.    ],
           [     -0.    ,       0.6481,       0.7615,       0.    ],
           [      1.    ,       0.    ,       0.    ,       0.    ],
           [   8842.5   ,  532069.326 ,  599608.6129,       1.    ]])

    In [96]: mm.T         ## have to transpose in order to use invert_homogenous
    Out[96]: 
    array([[      0.    ,      -0.    ,       1.    ,    8842.5   ],
           [      0.7615,       0.6481,       0.    ,  532069.326 ],
           [     -0.6481,       0.7615,       0.    ,  599608.6129],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [97]: invert_homogenous(mm.T)   ## invert homogenous assumes column vector layout, hence need to transform first 
    Out[97]: 
    array([[      0.    ,       0.7615,      -0.6481,  -16572.8991],
           [     -0.    ,       0.6481,       0.7615, -801469.6472],
           [      1.    ,       0.    ,       0.    ,   -8842.5   ],
           [      0.    ,       0.    ,       0.    ,       1.    ]])

    In [98]: g = np.dot( invert_homogenous(mm.T), [0,0,0,1] )

    In [99]: g
    Out[99]: array([ -16572.8991, -801469.6472,   -8842.5   ,       1.    ])


    In [100]: np.dot( mm.T, g ) # post-mult 
    Out[100]: array([ 0., -0., -0.,  1.])

    In [101]: np.dot( mm , g )   # nope
    Out[101]: array([ -6.0462e+05,  -5.2618e+05,  -1.6573e+04,  -4.3189e+11])

    In [102]: np.dot( g, mm )    # pre-mult
    Out[102]: array([ 0., -0., -0.,  1.])





"""

import logging, sys
log = logging.getLogger(__name__)


import IPython
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




