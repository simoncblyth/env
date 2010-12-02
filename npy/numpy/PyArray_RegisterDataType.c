/*
*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>



int main(int argc, char *argv[])
{

     Py_Initialize();
     import_array();

     PyObject *op ;
     op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     PyObject_Print( op , stdout, 0);

     PyArray_Descr* descr;
     PyArray_DescrConverter(op, &descr);
     Py_DECREF(op);
     
     int type = PyArray_RegisterDataType( descr );
     printf("type %d\n", type );

     PyArray_Descr* d ;
     d = PyArray_DescrFromType( type );
     PyObject_Print( (PyObject*)d , stdout, 0);


     return 0;
}



