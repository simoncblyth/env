/*

  Simplified creation of NumPy arrays from pre-allocated memory.

  such that they get deallocated when the numpy array is 

   http://blog.enthought.com/page/5/
   http://blog.enthought.com/page/39/

*/


#include "aligned.h"


int nd=2;
npy_intp dims[2]={10,20};
size_t size;
PyObject arr=NULL;
void *mymem;

size = PyArray_MultiplyList(dims, nd);
mymem = _aligned_malloc(size, 16);
arr = PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, mymem);
if (arr == NULL) {
    _aligned_free(mymem);
    Py_XDECREF(arr);
}
PyArray_BASE(arr) = PyCObject_FromVoidPtr(mymem, _aligned_free);






