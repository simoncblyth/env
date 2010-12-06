/*
    int PyArray_DescrConverter(PyObject* obj, PyArray_Descr** dtype)

    Convert any compatible Python object, obj, to a data-type object in dtype. A large number of Python objects can be converted to data-type objects. See Data type objects (dtype) for a complete description. This version of the converter converts None objects to a NPY_DEFAULT_TYPE data-type object. This function can be used with the ?O&? character code in PyArg_ParseTuple processing.

int PyArray_DescrConverter2(PyObject* obj, PyArray_Descr** dtype)

    Convert any compatible Python object, obj, to a data-type object in dtype. This version of the converter converts None objects so that the returned data-type is NULL. This function can also be used with the ?O&? character in PyArg_ParseTuple processing.

int Pyarray_DescrAlignConverter(PyObject* obj, PyArray_Descr** dtype)

    Like PyArray_DescrConverter except it aligns C-struct-like objects on word-boundaries as the compiler would.

int Pyarray_DescrAlignConverter2(PyObject* obj, PyArray_Descr** dtype)

    Like PyArray_DescrConverter2 except it aligns C-struct-like objects on word-boundaries as the compiler would.


*/



#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>


int main(int argc, char *argv[])
{

     Py_Initialize();
     import_array();

     //PyObject* op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     PyObject* op = Py_BuildValue("{s:(si),s:(si)}", "aaaa", "i4",0, "bbbb", "f4",4 );
     PyObject_Print( op , stdout, 0);
     printf("\n");

     PyArray_Descr* dtype ;
     PyArray_DescrConverter(op, &dtype);
     //PyArray_DescrAlignConverter(op, &dtype );
     Py_DECREF(op);

     PyObject_Print( (PyObject*)dtype , stdout, 0);

     printf("\n");

     return 0 ;
}    



