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
     "NPY_BYTE", 
     "NPY_UBYTE",
     "NPY_SHORT", 
     "NPY_USHORT",
     "NPY_INT", 
     "NPY_UINT",
     "NPY_LONG", 
     "NPY_ULONG",
     "NPY_LONGLONG", 
     "NPY_ULONGLONG",
     "NPY_HALF", 
     "NPY_FLOAT", 
     "NPY_DOUBLE", 
     "NPY_LONGDOUBLE",
     "NPY_CFLOAT", 
     "NPY_CDOUBLE", 
     "NPY_CLONGDOUBLE",
     "NPY_DATETIME", 
     "NPY_TIMEDELTA",
     "NPY_OBJECT",
     "NPY_STRING", 
     "NPY_UNICODE",
     "NPY_VOID" } ;
 
char* NPY_TYPE_FMTS[NPY_NTYPES] =  { 
     "%"NPY_BYTE_FMT, // "NPY_BOOL_FMT",
     "%"NPY_BYTE_FMT, 
     "%"NPY_UBYTE_FMT ,
     "%"NPY_SHORT_FMT, 
     "%"NPY_USHORT_FMT,
     "%"NPY_INT_FMT, 
     "%"NPY_UINT_FMT,
     "%"NPY_LONG_FMT, 
     "%"NPY_ULONG_FMT,
     "%"NPY_LONGLONG_FMT, 
     "%"NPY_ULONGLONG_FMT,
     "%"NPY_HALF_FMT, 
     "%"NPY_FLOAT_FMT, 
     "%lg",  // "%"NPY_DOUBLE_FMT, 
     "%Lg", // %"NPY_LONGDOUBLE_FMT,
     "NPY_CFLOAT_FMT", 
     "NPY_CDOUBLE", 
     "NPY_CLONGDOUBLE",
     "%"NPY_DATETIME_FMT, 
     "%"NPY_TIMEDELTA_FMT,
     "NPY_OBJECT_FMT",
     "NPY_STRING_FMT", 
     "NPY_UNICODE_FMT",
     "NPY_VOID_FMT" } ;



void bitsof()
{
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
}

void printf_formats()
{
    npy_bool bool_ = 1 ;
    npy_byte byte  = 1 ; 
    npy_ubyte ubyte = 1 ;
    npy_short short_ = 1 ;
    npy_ushort ushort_ = 1 ;
    npy_int int_ = 1 ;
    npy_uint uint = 1 ;
    npy_long long_ = 1 ;
    npy_ulong ulong = 1 ;
    npy_longlong longlong  = 1 ;
    npy_ulonglong ulonglong  = 1 ;

    printf("byte %hhd\n",byte ) ;


    npy_half half = 1 ;
    npy_float float_ = 1 ;
    npy_double double_ = 1 ;
    npy_longdouble longdouble = 1 ;

//    npy_cfloat cfloat = 1 ;
//    npy_cdouble cdouble = 1 ;
//    npy_clongdouble clongdouble = 1 ;
    
    char* str = "123" ;

    sscanf( str , "%hhg" , &half );
    sscanf( str , "%hg" ,  &float_ );
    sscanf( str , "%lg" ,  &double_ );
    sscanf( str , "%Lg" ,  &longdouble );
 
    printf("half %hhg \n", half );
    printf("float %hg \n", float_ );
    printf("double %lg \n", double_ );
    printf("longdouble %Lg \n", longdouble );
   
  

}
/*
'NPY_BOOL' ---> (dtype('bool'), 0) type_num 0 type ? kind b elsize 1 offset 0 fmt NPY_BOOL_FMT NPY_BOOL_FMT rc 0 
'NPY_BYTE' ---> (dtype('int8'), 1) type_num 1 type b kind i elsize 1 offset 1 fmt %hhd -55 rc 1 
'NPY_UBYTE' ---> (dtype('uint8'), 2) type_num 2 type B kind u elsize 1 offset 2 fmt %hhu 202 rc 1 
'NPY_SHORT' ---> (dtype('int16'), 3) type_num 3 type h kind i elsize 2 offset 3 fmt %hd 9163 rc 1 
'NPY_USHORT' ---> (dtype('uint16'), 5) type_num 4 type H kind u elsize 2 offset 5 fmt %hu 9165 rc 1 
'NPY_INT' ---> (dtype('int32'), 7) type_num 7 type l kind i elsize 4 offset 7 fmt %ld 168436687 rc 1 
'NPY_UINT' ---> (dtype('uint32'), 11) type_num 8 type L kind u elsize 4 offset 11 fmt %lu 168436691 rc 1 
'NPY_LONG' ---> (dtype('int32'), 15) type_num 7 type l kind i elsize 4 offset 15 fmt %ld 168436695 rc 1 
'NPY_ULONG' ---> (dtype('uint32'), 19) type_num 8 type L kind u elsize 4 offset 19 fmt %lu 168436699 rc 1 
'NPY_LONGLONG' ---> (dtype('int64'), 23) type_num 9 type q kind i elsize 8 offset 23 fmt %Ld 723430130999501791 rc 1 
'NPY_ULONGLONG' ---> (dtype('uint64'), 31) type_num 10 type Q kind u elsize 8 offset 31 fmt %Lu 723430165359240167 rc 1 
'NPY_HALF' ---> (dtype('float16'), 39) type_num 11 type e kind f elsize 2 offset 39 fmt %g 2.65647e-260 rc 1 
'NPY_FLOAT' ---> (dtype('float32'), 41) type_num 12 type f kind f elsize 4 offset 41 fmt %g 2.65648e-260 rc 1 
'NPY_DOUBLE' ---> (dtype('float64'), 45) type_num 13 type d kind f elsize 8 offset 45 fmt %g 2.65648e-260 rc 1 
'NPY_LONGDOUBLE' ---> (dtype('float96'), 53) type_num 14 type g kind f elsize 12 offset 53 fmt %Lg 1.68771e-4931 rc 1 
'NPY_CFLOAT' ---> (dtype('complex64'), 65) type_num 15 type F kind c elsize 8 offset 65 fmt NPY_CFLOAT_FMT NPY_CFLOAT_FMT rc 0 
'NPY_CDOUBLE' ---> (dtype('complex128'), 73) type_num 16 type D kind c elsize 16 offset 73 fmt NPY_CDOUBLE NPY_CDOUBLE rc 0 
'NPY_CLONGDOUBLE' ---> (dtype('complex192'), 89) type_num 17 type G kind c elsize 24 offset 89 fmt NPY_CLONGDOUBLE NPY_CLONGDOUBLE rc 0 
*/

void dump_dtype( PyArray_Descr* dtype )
{

    PyObject_Print( (PyObject*)dtype , stdout, 0 );
    printf("\n");

    PyObject *key, *tup ;
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

}

 
int main(int argc, char *argv[])
{
     Py_Initialize();
     import_array();

     printf_formats();
     PyObject* l = PyList_New(0) ;
     PyArray_Descr* d ;
     char code[10];

     int t ;
     for( t=0 ; t < NPY_NTYPES  ; t++ ){
         int is_flexible = PyTypeNum_ISFLEXIBLE(t) ;
         d = is_flexible ? PyArray_DescrNewFromType(t) : PyArray_DescrFromType( t );
         if(is_flexible){
            d->elsize = 7 ; 
         }
         //sprintf( code, "%c%d", d->type , d->elsize );
         sprintf( code, "%c%d", d->kind , d->elsize );
         printf("typenum %2d flexible %d name %20s fmt %20s kind %c type %c byteorder %c elsize %2d code %s ", t, is_flexible, NPY_TYPE_NAMES[t],NPY_TYPE_FMTS[t], d->kind , d->type ,d->byteorder,  d->elsize, code  );

         PyObject_Print( (PyObject*)d , stdout, 0 );
         printf("\n");

         //if( t == NPY_SHORT || t == NPY_INT || t == NPY_UINT || t == NPY_LONG || t == NPY_ULONG || t == NPY_FLOAT || t == NPY_DOUBLE ){
         //if( PyTypeNum_ISINTEGER(t) ){ 
         //if( PyTypeNum_ISFLOAT(t) ){ 
         //if( PyTypeNum_ISNUMBER(t) ){ 
         // if( !PyTypeNum_ISCOMPLEX(t) && !PyTypeNum_ISFLEXIBLE(t) && t != NPY_OBJECT ){ 
         
         if( !PyTypeNum_ISOBJECT(t) ){
              
              PyObject* op = Py_BuildValue("(s,s)", NPY_TYPE_NAMES[t] , code );
              PyList_Append( l , op );
         }
    }

    printf("dtype blueprint ... \n");
    PyObject_Print( (PyObject*)l , stdout, 0 );
    printf("\n");

    PyArray_Descr *dtype;

    PyArray_DescrConverter( l , &dtype );
    //dump_dtype( dtype );

    //PyArray_DescrAlignConverter( l , &dtype );
    //dump_dtype( dtype );


    // manual buffer filling 
    size_t size = dtype->elsize*1 ;
    void* data = malloc(size);

    PyObject *key, *tup ;
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
                       printf(" type_num %d type %c kind %c elsize %d offset %d fmt %s ", type_num, fdt->type, fdt->kind, fdt->elsize, offset, fmt );


                       void* ptr = data + offset ;  

                       int rc ;
                       char* str = "123" ;
                       rc = sscanf( str,  fmt , ptr ) ;

                       switch( type_num ){              
                            case NPY_BOOL:       printf( fmt , *((npy_bool*)ptr))       ; break ;
                            case NPY_BYTE:       printf( fmt , *((npy_byte*)ptr))       ; break ;
                            case NPY_UBYTE:      printf( fmt , *((npy_ubyte*)ptr))      ; break ;
                            case NPY_SHORT:      printf( fmt , *((npy_short*)ptr))      ; break ;
                            case NPY_USHORT:     printf( fmt , *((npy_ushort*)ptr))     ; break ;
                            case NPY_INT:        printf( fmt , *((npy_int*)ptr))        ; break ;
                            case NPY_UINT:       printf( fmt , *((npy_uint*)ptr))       ; break ;
                            case NPY_LONG:       printf( fmt , *((npy_long*)ptr))       ; break ;
                            case NPY_ULONG:      printf( fmt , *((npy_ulong*)ptr))      ; break ;
                            case NPY_LONGLONG:   printf( fmt , *((npy_longlong*)ptr))   ; break ;
                            case NPY_ULONGLONG:  printf( fmt , *((npy_ulonglong*)ptr))  ; break ;
                            case NPY_HALF:       printf( fmt , *((npy_half*)ptr))       ; break ;
                            case NPY_FLOAT:      printf( fmt , *((npy_float*)ptr))      ; break ;
                            case NPY_DOUBLE:     printf( fmt , *((npy_double*)ptr))     ; break ;
                            case NPY_LONGDOUBLE: printf( fmt , *((npy_longdouble*)ptr)) ; break ;
                            case NPY_CFLOAT:     printf( fmt , *((npy_cfloat*)ptr))     ; break ;
                            case NPY_CDOUBLE:    printf( fmt , *((npy_cdouble*)ptr))    ; break ;
                            case NPY_CLONGDOUBLE:printf( fmt , *((npy_clongdouble*)ptr)); break ;
                            case NPY_DATETIME:   printf( fmt , *((npy_datetime*)ptr))   ; break ;
                            case NPY_TIMEDELTA:  printf( fmt , *((npy_timedelta*)ptr))  ; break ;
                            //case NPY_OBJECT:      sscanf( str,  fmt ,      (npy_object*)ptr) ; break ;
                            //case NPY_STRING:      sscanf( str,  fmt ,      (npy_string*)ptr) ; break ;
                            //case NPY_UNICODE:     sscanf( str,  fmt ,     (npy_unicode*)ptr) ; break ;
                            //case NPY_VOID:        sscanf( str,  fmt ,        (npy_void*)ptr) ; ;break;
                       }  // type switch

                       printf( " rc %d \n", rc ) ;
                   } // tuple length check
             } // tuple check
         }    // over field names
     }       // names and field check 

 
    PyObject* buf = PyBuffer_FromMemory( data , (Py_ssize_t)size) ;
    PyObject_Print( (PyObject*)buf , stdout, 0);
    printf("\n");
    
    npy_intp count = 1 ;
    npy_intp bufoff = 0 ;

    PyObject* a = PyArray_FromBuffer( buf, dtype , count, bufoff);
    printf(" a %x \n", a );

    PyObject_Print( (PyObject*)a , stdout, 0);
    printf("\n");



    return 0;
}



