#!/usr/bin/env python

import sys, numpy as np

try:
    import opticks.ana.search as search
except ImportError:
    search = None
pass

if search is None:
    print("failed to import f2py opticks.ana.search module")
    print("try with PYTHONPATH=$OPTICKS_PREFIX/py " )
    sys.exit(1)
pass

print(search.find_first.__doc__)

q = np.random.randint(0, 1000000, (1000000,) )
qu = np.unique(q)

print("q",q)
print("q.shape",str(q.shape))
print("qu",qu)
print("qu.shape",str(qu.shape))


idx_expected = len(qu)//2   ## pick idx in middle
v = qu[idx_expected]       ## random value from middle
idx = search.find_first(v,qu)  

print("idx_expected %d v %d idx %d " % ( idx_expected, v, idx ))

assert idx == idx_expected









