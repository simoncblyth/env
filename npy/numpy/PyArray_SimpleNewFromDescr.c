/*

    EXPLORE CREATION AND FILLING OF NUMPY ARRAYS FROM  C
    WITH EYE TO TURNING MYSQL_ ROW query 
    result directly into numpy structured array

  http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
  http://docs.scipy.org/doc/numpy/reference/c-api.html
  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#from-scratch 

  http://svn.scipy.org/svn/scipy/branches/testing_cleanup/scipy/sandbox/timeseries/src/c_tseries.c
  http://starship.python.net/crew/hinsen/NumPyExtensions.html 

  http://www.pymolwiki.org/index.php/Advanced_Scripting

*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#include "mysql.h"
#include "mysql_np.h"


int main(int argc, char *argv[])
{
     int dims[] = { 2, 3 };
     PyObject *op, *array, *dte ;
     PyArray_Descr *descr;

     Py_Initialize();
     import_array();

 
     // shortest but least flexible ...  
     //op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     
     // more easily extendible
     //op = PyList_New(2) ;
     //dte = Py_BuildValue("(s,s)" , "aaa", "i4" ) ;
     //PyList_SET_ITEM( op, 0, dte );
     //dte = Py_BuildValue("(s,s)" , "bbb", "f4" ) ;
     //PyList_SET_ITEM(op, 1, dte );

     // list appending style is most convenient
     op = PyList_New(0) ;
     Py_INCREF(op);
     PyObject* kv ;
     int ikv ;
     for ( ikv = 0; ikv < 2; ikv++ ) {
         if( ikv == 0) kv = Py_BuildValue( "(s,s)", "aaa", "i4" );
         if( ikv == 1) kv = Py_BuildValue( "(s,s)", "bbb", "f4" );
         Py_INCREF(kv);
         PyList_Append(op, kv );
     }

     printf(" MYSQL_TYPE_TINY %d %s \n" , MYSQL_TYPE_TINY , mysql2np( MYSQL_TYPE_TINY) );
     printf(" MYSQL_TYPE_LONG %d %s \n" , MYSQL_TYPE_LONG , mysql2np( MYSQL_TYPE_LONG) );
     printf(" MYSQL_TYPE_FLOAT %d %s \n" , MYSQL_TYPE_FLOAT , mysql2np( MYSQL_TYPE_FLOAT) );
     printf(" MYSQL_TYPE_DOUBLE %d %s \n" , MYSQL_TYPE_DOUBLE ,mysql2np( MYSQL_TYPE_DOUBLE ) );
     printf(" MYSQL_TYPE_TIMESTAMP %d %s \n" , MYSQL_TYPE_TIMESTAMP , mysql2np(  MYSQL_TYPE_TIMESTAMP) );



     PyObject_Print( op , stdout, 0);
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



