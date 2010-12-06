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
         sprintf( code, "%c%d", d->type , d->elsize );
         printf("typenum %2d flexible %d name %20s fmt %20s kind %c type %c byteorder %c elsize %2d code %s ", t, is_flexible, NPY_TYPE_NAMES[t],NPY_TYPE_FMTS[t], d->kind , d->type ,d->byteorder,  d->elsize, code  );

         PyObject_Print( (PyObject*)d , stdout, 0 );
         printf("\n");

         PyObject* op = Py_BuildValue("(s,s)", NPY_TYPE_NAMES[t] , code );
         PyList_SET_ITEM( l , t, op );

    }

    PyObject_Print( (PyObject*)l , stdout, 0 );
    printf("\n");

    PyArray_Descr *dtype;
    PyArray_DescrConverter( l , &dtype );

    PyObject_Print( (PyObject*)dtype , stdout, 0 );
    printf("\n");

     PyObject *key, *tup ;
     Py_ssize_t pos = 0;
     
     PyObject* names = dtype->names ;
     PyObject* fields = dtype->fields ;

    // use names in order to get the correct field order
     if(PyTuple_Check(names) && PyDict_Check(fields)){
          
         int n = PyTuple_GET_SIZE(names);
         int i ; 
         for ( i = 0; i < n; i++) {
              key = PyTuple_GET_ITEM(names, i);
              tup = PyDict_GetItem( fields, key);
              PyObject_Print( (PyObject*)key , stdout, 0);
              printf(" ---> ");
              PyObject_Print( (PyObject*)tup , stdout, 0);

             if(PyTuple_Check(tup)){
                   Py_ssize_t stup = PyTuple_GET_SIZE(tup);
                   if( (int)stup >= 2 ){
                       PyObject* ft  = PyTuple_GetItem( tup , (Py_ssize_t)0 );
                       PyObject* fo  = PyTuple_GetItem( tup , (Py_ssize_t)1 );
                       PyArray_Descr* fdt = (PyArray_Descr*)ft ;
                       int type_num = fdt->type_num ;
                       char* fmt = NPY_TYPE_FMTS[type_num] ;  
                       int offset   =  (int)PyLong_AsLong( fo ); 
                       printf(" type_num %d type %c kind %c elsize %d offset %d fmt %s \n", type_num, fdt->type, fdt->kind, fdt->elsize, offset, fmt );
                   }
             }


         }

     }



    return 0;
}



