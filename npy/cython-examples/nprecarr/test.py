#
# ----- REMEMBER TO COMPILE AFTER ANY CHANGE TO .pyx ----
#
#        rm -rf test_recarray.c test_recarray.so build ; python setup.py build_ext -i && python test.py
#
#

import numpy as np
import recarray
import traceback

dt0 = np.dtype([('f0', 'f4'), ('i0', 'i4'), ('i1', 'i4')])
dt1 = np.dtype([('f0', 'f4'), ('i0', 'i4'), ('i1', 'i4'),('j0','i4')])
dt2 = np.dtype([('db', 'f4'), ('k', 'i4'), ('i0', 'i4'), ('i1', 'i4'),('j0', 'i4'), ('j1', 'i4')])

ar0 = np.empty(shape=(5,5), dtype = dt0)       ## gibberish numbers
ar0['f0'] = np.random.random((5,5))     
ar0['i0'] = 0                         ## all elements of the 5x5 set to 0 
ar0['i1'] = 10

try:
  recarray.test0(ar0)
  print "test0 success"
except:
  print "test0 fail"
  traceback.print_exc()

ar1 = np.empty(shape=(5,5), dtype = dt1)
ar1['f0'] = np.random.random((5,5))
ar1['i0'] = 0
ar1['i1'] = 10
ar1['j0'] = 20

try:
  recarray.test1(ar1)
  print "test1 success"
except:
  print "test1 fail"
  traceback.print_exc()

ar2 = np.empty(shape=(5,5), dtype = dt2)
ar2['db'] = np.random.random((5,5))
ar2['k'] = -1
ar2['i0'] = 0
ar2['i1'] = 10
ar2['j0'] = 20
ar2['j1'] = 30

try:
  recarray.test2( ar2)
  print "test2 success"
except:
  print "test2 fail"
  traceback.print_exc()

print ar2



ar3 = np.empty(shape=(5,5), dtype = dt2)
ar3['db'] = np.random.random((5,5))
ar3['k'] = 1
ar3['i0'] = 2
ar3['i1'] = 3
ar3['j0'] = 4
ar3['j1'] = 5

try:
  recarray.test3( ar3)
  print "test3 success"
except:
  print "test3 fail"
  traceback.print_exc()


print ar3


ar4 = np.empty(shape=(5,5), dtype = dt2)
ar4['db'] = np.random.random((5,5))
ar4['k'] = 1
ar4['i0'] = 2
ar4['i1'] = 3
ar4['j0'] = 4
ar4['j1'] = 5

try:
  recarray.test4( ar4)
  print "test4 success"
except:
  print "test4 fail"
  traceback.print_exc()


print ar4





