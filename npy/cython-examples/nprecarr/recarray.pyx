#  google:"cython numpy recarray"
# http://permalink.gmane.org/gmane.comp.python.cython.user/353
# test_recarray.pyx

import numpy as np
cimport numpy as np



# https://github.com/dagss/euroscipy2010/blob/master/gsl/spline.pyx
from libc.stdlib cimport const_char, strtof



cdef packed struct rec_cell0:
  np.float32_t f0
  np.int32_t i0, i1

cdef packed struct rec_cell1:
  np.float32_t f0
  np.int32_t i0, i1, j0

cdef packed struct rec_cell2:
  np.float32_t db
  np.int32_t k, i0, i1, j0, j1


def test0(np.ndarray[rec_cell0, ndim=2] recarray):
  cdef Py_ssize_t i,j
  cdef rec_cell0 *cell
  for i in range(recarray.shape[0]):
    for j in range(recarray.shape[1]):
      cell = &recarray[i,j]
      print "f: %f i: %d %d"%(cell.f0, cell.i0, cell.i1)

def test1(np.ndarray[rec_cell1, ndim=2] recarray):
  cdef Py_ssize_t i,j
  cdef rec_cell1 *cell
  for i in range(recarray.shape[0]):
    for j in range(recarray.shape[1]):
      cell = &recarray[i,j]
      print "f: %f i: %d %d %d"%(cell.f0, cell.i0, cell.i1, cell.j0)


# this way succeeds to modify the recarray in place ... so the calling python sees the changes 
def test2(np.ndarray[rec_cell2, ndim=2] recarray):
  cdef Py_ssize_t i,j
  cdef rec_cell2 *cell
  for i in range(recarray.shape[0]):
    for j in range(recarray.shape[1]):
      print i,j 
      cell = &recarray[i,j]
      print(" i0 %d " % cell.i0 ) 
      print "f: %f k: %d i: %d %d %d %d"%(cell.db, cell.k, cell.i0, cell.i1, cell.j0, cell.j1)
      cell.k = i + j 
      cell.i0 = 100
      cell.i1 = 200
      cell.j0 = 300
      cell.j1 = 400 

def test3(np.ndarray[rec_cell2, ndim=2] recarray):
  cdef Py_ssize_t i,j
  cdef rec_cell2 rec
  for i in range(recarray.shape[0]):
    for j in range(recarray.shape[1]):
      rec = recarray[i,j]
      print "f: %f k: %d i: %d %d %d %d"%(rec.db, rec.k, rec.i0, rec.i1, rec.j0, rec.j1)
      rec.k = i + j 
      rec.i0 = 100
      rec.i1 = 200
      rec.j0 = 300
      rec.j1 = 400 


def test4(np.ndarray[rec_cell2, ndim=2] recarray):
  cdef Py_ssize_t i,j
  cdef char*  endp = NULL 
  cdef const_char* str = "123456.012" 


  for i in range(recarray.shape[0]):
    for j in range(recarray.shape[1]):

      recarray[i,j].db = strtof( str , &endp )
      recarray[i,j].k = i + j 
      recarray[i,j].i0 = 100
      recarray[i,j].i1 = 200
      recarray[i,j].j0 = 300
      recarray[i,j].j1 = 400 


