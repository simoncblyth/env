
import numpy as np
cimport numpy as np   ## the numpy.pxd with cdefs etc... for using the numpy C lib

def myfunc(np.ndarray[np.float64_t, ndim=2] A): 
    cdef Py_ssize_t i, j 

    print np.NPY_INT8, np.NPY_INT16
    print A.descr

    for i in range(A.shape[0]): 
        print A[i, 0] # fast 
        j = 2*i 
        print A[i, j] # fast 
        k = 2*i 
        print A[i, k] # slow, k is not typed 
        print A[i][j] # slow ... is this fancy indexing ? 
        print A[i,:] # slow 

