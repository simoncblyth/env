/*

  NUMPY 2.0.0 DEBUGGING SESSION ... 
     * PyObject_Print of numpy structured areas created from external buffer was segmenting  
     * problem in scalar ctor ... userdefined types were not treated as flexible

      TODO : REPORT TO NUMPY TRAC 




  APPEARING BE FIXED WITH 

[blyth@cms01 numpy]$ git diff
diff --git a/numpy/core/src/multiarray/scalarapi.c b/numpy/core/src/multiarray/scalarapi.c
index 87e140c..0f84d87 100644
--- a/numpy/core/src/multiarray/scalarapi.c
+++ b/numpy/core/src/multiarray/scalarapi.c
@@ -674,7 +674,7 @@ PyArray_Scalar(void *data, PyArray_Descr *descr, PyObject *base)
         memcpy(&(((PyDatetimeScalarObject *)obj)->obmeta), dt_data,
                sizeof(PyArray_DatetimeMetaData));
     }
-    if (PyTypeNum_ISFLEXIBLE(type_num)) {
+    if (PyTypeNum_ISEXTENDED(type_num)) {
         if (type_num == PyArray_STRING) {
             destptr = PyString_AS_STRING(obj);
             ((PyStringObject *)obj)->ob_shash = -1;



*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "aligned.h"

int main(int argc, char *argv[])
{

     Py_Initialize();
     import_array();

     PyObject *op ;
     op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     PyObject_Print( op , stdout, 0);
     printf("\n");

     PyArray_Descr* descr;
     PyArray_DescrConverter(op, &descr);
     Py_DECREF(op);

     PyObject *key, *value;
     Py_ssize_t pos = 0;
 
    if(PyDict_Check(descr->fields)){
         printf("print the descr->fields dict ...  %d\n", descr->fields );
         PyObject_Print( (PyObject*)descr->fields , stdout, 0);
         printf("\n\n");

         printf("iterate over the descr->fields dict ...  %d\n", descr->fields );
         while (PyDict_Next( descr->fields , &pos, &key, &value)) {
             PyObject_Print( (PyObject*)key , stdout, 0);
             printf(" ---> ");
             PyObject_Print( (PyObject*)value , stdout, 0);
             printf("\n");

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

         }

     } else {
         printf("descr->fields  not dict\n");
     }

    
     int type = PyArray_RegisterDataType( descr );
     printf("type %d\n", type );

     PyArray_Descr* d ;
     d = PyArray_DescrFromType( type );
   /*
    pulling from the register ... yields 
      dtype(('|V8', [('aaaa', '<i4'), ('bbbb', '<f4')]))
    with the dtype wanted embedded in there 


    */



     PyObject_Print( (PyObject*)d , stdout, 0 );

     printf("elsize %d\n", d->elsize );
     printf("fields %d\n", d->fields );


/*
     if(PyDict_Check(d->fields)){
         printf("print the fields dict ...  %d\n", d->fields );
         PyObject_Print( (PyObject*)d->fields , stdout, 0);
     } else {
         printf("not dict\n");
     }
     while (PyDict_Next( d->fields , &pos, &key, &value)) {
         PyObject_Print( (PyObject*)key , stdout, 0);
     }
*/

     npy_intp dims[1] ;
     int typenum = type ;
     int nd = 1 ;
     dims[0] = 3 ;


     // manual buffer filling 
     size_t size = PyArray_MultiplyList( dims , nd )*descr->elsize ; 
     void* data = malloc(size);
     //void* data = _aligned_malloc(size , 16 );

     //char* demo_str[6] = { "123", "123.0", "223", "223.0" , "323" , "323.0" } ; 

     int off = 0 ;
     sscanf( "123",    "%d",   (int*)(data + off)) ; off += sizeof(int)   ;       
     sscanf( "123.0",  "%f", (float*)(data + off)) ; off += sizeof(float) ;       
     sscanf( "223",    "%d",   (int*)(data + off)) ; off += sizeof(int)   ;       
     sscanf( "223.0",  "%f", (float*)(data + off)) ; off += sizeof(float) ;       
     sscanf( "323",    "%d",   (int*)(data + off)) ; off += sizeof(int)   ;       
     sscanf( "323.0",  "%f", (float*)(data + off)) ; off += sizeof(float) ;       

     int it ;
     off = 0 ;
     for( it = 0 ; it < dims[0]  ; it++ ){
          printf("it %d \n", it );
          int* outd = ((int*)data + off );
          printf(" outd : %d \n" , *outd );
          off += 1 ;

          float* outf = ((float*)data + off);
          printf(" outf : %f \n" , *outf );
          off += 1 ;
    }

    printf(" sizeof(void)   : %d \n" , sizeof(void) );  
    printf(" sizeof(char)   : %d \n" , sizeof(char) );  
    printf(" sizeof(int)   : %d \n" , sizeof(int) );  
    printf(" sizeof(float) : %d \n" , sizeof(float) );  
    printf(" sizeof(double) : %d \n" , sizeof(double) );  


    PyObject* buf ;
    buf = PyBuffer_FromMemory( data , (Py_ssize_t)size) ;
    PyObject_Print( (PyObject*)buf , stdout, 0);
    printf("\n");
    
    npy_intp count = 3 ;
    npy_intp bufoff = 0 ;

    //PyObject* a = PyArray_SimpleNewFromData( nd,  dims, typenum,  data) ;
    PyObject* a = PyArray_FromBuffer( buf, descr , count, bufoff);

  
    void* a_data = PyArray_DATA(a);
    npy_intp a_size = PyArray_Size(a) ;
    npy_intp a_nbytes = PyArray_NBYTES(a) ;
    int a_itemsize = PyArray_ITEMSIZE(a);
    int a_isaligned = PyArray_ISALIGNED(a);
    int a_isuserdef = PyArray_ISUSERDEF(a);
    int a_isflexible = PyArray_ISFLEXIBLE(a);  
    int a_isextended = PyArray_ISEXTENDED(a);  
    int a_isobject = PyArray_ISOBJECT(a);  
    int a_isnbo = PyArray_ISNBO(a);  
    PyArray_Descr* a_descr = PyArray_DESCR(a) ;


    printf(" a_size %d a_nbytes %d a_itemsize %d a_isaligned %d a_isuserdef %d a_isflexible %d a_isextended %d a_isobject %d a_isnbo %d \n", a_size, a_nbytes, a_itemsize, a_isaligned , a_isuserdef, a_isflexible, a_isextended, a_isobject, a_isnbo );
    printf("\n a_descr str \n");
    PyObject_Print( (PyObject*)a_descr , stdout, Py_PRINT_RAW );  // the str 
    printf("\n a_descr str done \n");
    printf("\n a_descr repr \n");
    PyObject_Print( (PyObject*)a_descr , stdout,  0 ); 
    printf("\n a_descr repr done \n");
       
    printf("\n data scalar 1\n");
    PyObject* ds = PyArray_ToScalar(a_data, a) ;      
    printf("\n data scalar 2\n");
    char* ds_obval = ((PyVoidScalarObject *)ds)->obval ;
    PyArray_Descr* ds_descr = ((PyVoidScalarObject *)ds)->descr ;  // tiz NULL

    printf("\n data scalar 3 %x %x \n", ds_obval, ds_descr );

    PyObject_Print(  ds , stdout, 0);   // SEGMENTS
    printf("\n data scalar 4\n");

    for( it = 0 ; it < count ; ++it ){
        void* ptr = PyArray_GETPTR1( a, (npy_intp)it ) ;
        printf(" from ptr %d ... %d \n " , it,  *((int*)ptr) );
        PyObject* item = PyArray_GETITEM(a , ptr) ;
        PyObject_Print( item , stdout, 0);
        printf("\n");
       
       /*   
        printf("\n item scalar 0\n");
        PyObject* sc = PyArray_ToScalar(ptr, a) ;   // SEGMENTS
        printf("\n item scalar 1\n");
        PyObject_Print(  sc , stdout, 0);
        printf("\n item scalar 2\n");
        */
    }

     

/*
  Program received signal SIGSEGV, Segmentation fault.
[Switching to Thread -1208640992 (LWP 28376)]
0x00ca46ef in LONG_copyswap (dst=0x0, src=0xb7f0aacc, swap=0, __NPY_UNUSED_TAGGEDarr=0x94f5ee0) at numpy/core/src/multiarray/arraytypes.c.src:1718
1718            memcpy(dst, src, sizeof(@type@));
(gdb) bt
#0  0x00ca46ef in LONG_copyswap (dst=0x0, src=0xb7f0aacc, swap=0, __NPY_UNUSED_TAGGEDarr=0x94f5ee0) at numpy/core/src/multiarray/arraytypes.c.src:1718
#1  0x00ca596f in VOID_copyswap (dst=0x0, src=0xb7f0aacc "\001", swap=0, arr=0x94f5ee0) at numpy/core/src/multiarray/arraytypes.c.src:2113
#2  0x00cdc873 in PyArray_Scalar (data=0xb7f0aacc, descr=0xb7c0c9f8, base=0x94f5ee0) at numpy/core/src/multiarray/scalarapi.c:776
#3  0x080494bd in main (argc=1, argv=0xbfe67fc4) at PyArray_SimpleNewFromData.c:170
(gdb) 


  #define PyArray_ToScalar(data, arr)                                           \
        PyArray_Scalar(data, PyArray_DESCR(arr), (PyObject *)arr)


     PyObject* PyArray_Scalar(void* data, PyArray_Descr* dtype, PyObject* itemsize)

    Return an array scalar object of the given enumerated typenum and itemsize by copying from memory pointed to by data .
    If swap is nonzero then this function will byteswap the data if appropriate to the data-type because array scalars are always in correct machine-byte order.


*/

   // from the segmenting ... VOID_copyswqp 

   /*

    if (PyArray_HASFIELDS(a)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new, *descr;
        int offset;
        Py_ssize_t pos = 0;

        descr = a_descr;
        while (PyDict_Next(descr->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                printf("descr field parse fails \n");
                return;
            }

            printf("\ndescr field type ... offset %d \n", offset);
            PyObject_Print(  key  , stdout, 0);
            PyObject_Print( (PyObject*)new , stdout, 0);
            PyObject_Print(  title , stdout, 0);
            printf("\ndescr field type done \n");
            
        }
    }

   */


    PyObject_Print( (PyObject*)a , stdout, Py_PRINT_RAW );  // the str 
    printf("\n");
    PyObject_Print( (PyObject*)a , stdout, 0);
    printf("\n");


    return 0;
}



