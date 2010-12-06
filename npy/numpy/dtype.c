/*

    http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes

   dtype.type 	The type object used to instantiate a scalar of this data-type.
   dtype.kind 	A character code (one of ?biufcSUV?) identifying the general kind of data.
   dtype.char 	A unique character code for each of the 21 different built-in types.
   dtype.num 	A unique number for each of the 21 different built-in types.
   dtype.str 	The array-protocol typestring of this data-type object.

   Size of the data is in turn described by:
   dtype.name 	A bit-width name for this data-type.
   dtype.itemsize 	The element size of this data-type object.


*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

char* NPY_TYPE_NAMES[NPY_NTYPES] =  { 
     "NPY_BOOL",
     "NPY_BYTE", "NPY_UBYTE",
     "NPY_SHORT", "NPY_USHORT",
     "NPY_INT", "NPY_UINT",
     "NPY_LONG", "NPY_ULONG",
     "NPY_LONGLONG", "NPY_ULONGLONG",
     "NPY_HALF", "NPY_FLOAT", "NPY_DOUBLE", "NPY_LONGDOUBLE",
     "NPY_CFLOAT", "NPY_CDOUBLE", "NPY_CLONGDOUBLE",
     "NPY_DATETIME", "NPY_TIMEDELTA",
     "NPY_OBJECT",
     "NPY_STRING", "NPY_UNICODE",
     "NPY_VOID" } ;
 
char* NPY_TYPE_FMTS[NPY_NTYPES] =  { 
     "NPY_BOOL_FMT",
     NPY_BYTE_FMT, NPY_UBYTE_FMT ,
     NPY_SHORT_FMT, NPY_USHORT_FMT,
     NPY_INT_FMT, NPY_UINT_FMT,
     NPY_LONG_FMT, NPY_ULONG_FMT,
     NPY_LONGLONG_FMT, NPY_ULONGLONG_FMT,
     NPY_HALF_FMT, NPY_FLOAT_FMT, NPY_DOUBLE_FMT, NPY_LONGDOUBLE_FMT,
     "NPY_CFLOAT_FMT", "NPY_CDOUBLE", "NPY_CLONGDOUBLE",
     NPY_DATETIME_FMT, NPY_TIMEDELTA_FMT,
     "NPY_OBJECT_FMT",
     "NPY_STRING_FMT", "NPY_UNICODE_FMT",
     "NPY_VOID_FMT" } ;
 
int main(int argc, char *argv[])
{

     Py_Initialize();
     import_array();

     printf(" NPY_BITSOF_SHORT      : %d \n", NPY_BITSOF_SHORT );
     printf(" NPY_BITSOF_INT        : %d \n", NPY_BITSOF_INT );
     printf(" NPY_BITSOF_LONG       : %d \n", NPY_BITSOF_LONG );
     printf(" NPY_BITSOF_LONGLONG   : %d \n", NPY_BITSOF_LONGLONG );
     printf("\n");
     printf(" NPY_BITSOF_FLOAT      : %d \n", NPY_BITSOF_FLOAT );
     printf(" NPY_BITSOF_DOUBLE     : %d \n", NPY_BITSOF_DOUBLE );
     printf(" NPY_BITSOF_LONGDOUBLE : %d \n", NPY_BITSOF_LONGDOUBLE );
     printf("\n");
     printf(" NPY_BITSOF_CHAR       : %d \n", NPY_BITSOF_CHAR );


     PyObject* l = PyList_New(NPY_NTYPES) ;
     PyArray_Descr* d ;
     char code[10];

     int t ;
     for( t=0 ; t < NPY_NTYPES  ; t++ ){
         int is_flexible = PyTypeNum_ISFLEXIBLE(t) ;
         d = is_flexible ? PyArray_DescrNewFromType(t) : PyArray_DescrFromType( t );
         if(is_flexible){
            d->elsize = 7 ; 
         }
         printf("typenum %2d flexible %d name %20s fmt %20s kind %c type %c byteorder %c elsize %2d ", t, is_flexible, NPY_TYPE_NAMES[t],NPY_TYPE_FMTS[t], d->kind , d->type ,d->byteorder,  d->elsize  );

         PyObject_Print( (PyObject*)d , stdout, 0 );
         printf("\n");

         sprintf( code, "%c%d", d->type , d->elsize );
         PyObject* op = Py_BuildValue("(s,s)", NPY_TYPE_NAMES[t] , code );
         PyList_SET_ITEM( l , t, op );

    }

    PyObject_Print( (PyObject*)l , stdout, 0 );
    printf("\n");

    PyArray_Descr *dtype;
    PyArray_DescrConverter( l , &dtype );

    PyObject_Print( (PyObject*)dtype , stdout, 0 );
    printf("\n");

     PyObject *key, *value;
     Py_ssize_t pos = 0;
 
     if(PyDict_Check(dtype->fields)){
         PyObject_Print( (PyObject*)dtype->fields , stdout, 0);
         printf("\n\n");

         while (PyDict_Next( dtype->fields , &pos, &key, &value)) {
             PyObject_Print( (PyObject*)key , stdout, 0);
             printf(" ---> ");
             PyObject_Print( (PyObject*)value , stdout, 0);
             printf("\n");

             /*
             if(PyTuple_Check(value)){
                   Py_ssize_t s = PyTuple_GET_SIZE(value);
                   if( (int)s >= 2 ){
                       
                       PyObject* ft  = PyTuple_GetItem( value, (Py_ssize_t)0 );
                       printf(" field type ... elsize %d \n", ((PyArray_Descr*)ft)->elsize );
                       PyObject_Print( ft , stdout, 0);
                       printf("\n");

                       PyObject* fo = PyTuple_GetItem( value, (Py_ssize_t)1 );
                       int ifo = (int)PyLong_AsLong( fo );
                       printf(" field offset ... %d ", ifo );

                       PyObject_Print( fo , stdout, 0);
                       printf("\n");
                   }
             }
             */

         }
     }

 

    return 0;
}



