/*

    EXPLORE CREATION AND FILLING OF NUMPY ARRAYS FROM  C
    WITH EYE TO TURNING MYSQL_ ROW query 
    result directly into numpy structured array

  http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
  http://docs.scipy.org/doc/numpy/reference/c-api.html
  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#from-scratch 

  http://svn.scipy.org/svn/scipy/branches/testing_cleanup/scipy/sandbox/timeseries/src/c_tseries.c
  http://starship.python.net/crew/hinsen/NumPyExtensions.html 

*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

int main(int argc, char *argv[])
{
     int dims[] = { 2, 3 };
     PyObject *op, *array ;
     PyArray_Descr *descr;

     Py_Initialize();
     import_array();

     op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     PyArray_DescrConverter(op, &descr);
     Py_DECREF(op);
     
     //array = PyArray_SimpleNewFromDescr(2, dims, descr);
     array = PyArray_Zeros(2, dims, descr, 0);

     int ix,iy ;
     for (ix = 0 ; ix < dims[0] ; ++ix ){
     for (iy = 0 ; iy < dims[1] ; ++iy ){

        npy_intp i = (npy_intp)ix ;
        npy_intp j = (npy_intp)iy ;
        void* ptr = PyArray_GETPTR2(array, i , j ) ;
    
        //PyObject* val = PyInt_FromLong(7L);                // wrong shape the setting fails
        //PyObject* val = Py_BuildValue("(ii)", 7, 8 );
        PyObject* val = Py_BuildValue("(ss)", "7", "8" );   // numpy does the needed conversions

        PyObject_Print(val , stdout, 0);
        printf("\n");
        int rc = PyArray_SETITEM(array, ptr, val);
     }
     }

     PyObject_Print(array, stdout, 0);
     printf("\n");
     Py_DECREF(array);
     return 0;
}



