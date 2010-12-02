/*
    Adapted from 
        http://cms01.phys.ntu.edu.tw/np/user/c-info.python-as-glue.html

   sys:1: DeprecationWarning: PyArray_FromDims: use PyArray_SimpleNew.
   sys:1: DeprecationWarning: PyArray_FromDimsAndDataAndDescr: use PyArray_NewFromDescr.

*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
 
PyObject* demo_add(PyObject* a_, PyObject* b_){
  
  PyArrayObject* a=(PyArrayObject*) a_;
  PyArrayObject* b=(PyArrayObject*) b_;
  int n = a->dimensions[0];
  int dims[1];
  dims[0] = n;
  PyArrayObject* ret;
  ret = (PyArrayObject*) PyArray_FromDims(1, dims, NPY_DOUBLE);
  int i;
  char *aj=a->data;
  char *bj=b->data;
  double *retj = (double *)ret->data;
  for (i=0; i < n; i++) {
    *retj++ = *((double *)aj) + *((double *)bj);
    aj += a->strides[0];
    bj += b->strides[0];
  }
  return (PyObject *)ret;
}


int main(int argc, char *argv[])
{

   Py_Initialize();
   import_array();

   double start, stop, step ;
   start = 0. ;
   stop = 10. ;
   step = 1.  ;

   PyArrayObject* aa ;
   aa = (PyArrayObject*)PyArray_Arange(start, stop, step, NPY_DOUBLE); 
   
   PyArrayObject* bb ;
   bb = (PyArrayObject*)PyArray_Arange(start, stop, step, NPY_DOUBLE);

   PyObject* d ;
   d = demo_add( (PyObject*)aa, (PyObject*)bb ) ;
   PyObject_Print( d , stdout, 0);
   
   return 0 ;
}





