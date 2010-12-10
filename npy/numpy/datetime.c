/*
*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>


void po( const char* msg , PyObject* obj ){
     printf("%s\n", msg);
     PyObject_Print( obj , stdout, 0);
     printf("\n");
}


int main(int argc, char *argv[])
{
     npy_intp dims[] = { 5 };

     Py_Initialize();
     import_array();

     PyObject *op ;
     op = Py_BuildValue("[(s,s),(s,s),(s,s)]", "a_int", "i4", "a_float", "f4", "a_datetime", "M8[s]" );
     po("op", op );
     
     PyArray_Descr* descr;
     PyArray_DescrConverter(op, &descr);
     po("descr", descr );

     PyObject* array ;
     array = PyArray_SimpleNewFromDescr(1 , dims, descr);

     npy_intp n = dims[0] ; 
     npy_intp i ;
     for (i = 0 ; i < n ; ++i ){
        void* ptr = PyArray_GETPTR1(array, (npy_intp)i ) ;

        PyObject* tup = PyTuple_New(3);
        PyTuple_SET_ITEM( tup , 0 , PyString_FromString( "1" ) );
        PyTuple_SET_ITEM( tup , 1 , PyString_FromString( "2" ) );
        PyTuple_SET_ITEM( tup , 2 , PyString_FromString( "3" ) );

        int rc = PyArray_SETITEM( array, ptr, tup );
        po("tup", tup );
     }

     po("array", array);


     return 0;
}



