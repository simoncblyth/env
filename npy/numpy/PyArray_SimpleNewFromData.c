/*
*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

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



     PyObject_Print( (PyObject*)d , stdout, 0);

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

     int offset = 0 ;
     sscanf( "123",    "%d",     ((int*)data + offset)) ; offset += 1 ; //sizeof(int) ;       
     sscanf( "123.0",  "%f",   ((float*)data + offset)) ; offset += 1 ; //sizeof(float) ;       
     sscanf( "223",    "%d",     ((int*)data + offset)) ; offset += 1 ; //sizeof(int) ;       
     sscanf( "223.0",  "%f",   ((float*)data + offset)) ; offset += 1 ; //sizeof(float) ;       
     sscanf( "323",    "%d",     ((int*)data + offset)) ; offset += 1 ; //sizeof(int) ;       
     sscanf( "323.0",  "%f",   ((float*)data + offset)) ; offset += 1 ; //sizeof(float) ;       

     int it ;
     int off = 0 ;
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
    
    npy_intp a_size = PyArray_Size(a) ;
    npy_intp a_nbytes = PyArray_NBYTES(a) ;
    int a_itemsize = PyArray_ITEMSIZE(a);
    printf(" a_size %d a_nbytes %d a_itemsize %d \n", a_size, a_nbytes, a_itemsize );

    for( it = 0 ; it < count ; ++it ){
        void* ptr = PyArray_GETPTR1( a, (npy_intp)it ) ;
        printf(" from ptr %d ... %d \n " , it,  *((int*)ptr) );
        PyObject* item = PyArray_GETITEM(a , ptr) ;
        PyObject_Print( item , stdout, 0);
        printf("\n");
    }     


    PyObject_Print( (PyObject*)a , stdout, 0);
    printf("\n");


    return 0;
}



