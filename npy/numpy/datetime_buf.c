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

#define NF 4 

int main(int argc, char *argv[])
{
     Py_Initialize();
     import_array();

     // manual approach for debugging 
     int offsets[NF] = { 0, 4, 8 , 16 } ;
     char fmts[NF][10] = { "%d" , "%g" , "%lld" , "%Ld" } ; 
     PyObject* op = Py_BuildValue("[(s,s),(s,s),(s,s),(s,s)]", "a_int", "i4", "a_float", "f4", "a_datetime", "M8[s]" , "b_datetime", "M8[s]" );
     po("op", op );
    
     PyArray_Descr* descr;
     PyArray_DescrConverter(op, &descr);
     po("descr", descr );

     npy_intp count = 2 ;
     size_t size = descr->elsize*count ;
     void* data = malloc(size);
     void* rec = data ;

     int try ;
     for( try=0 ; try < 2 ; try++ ){
        char* str ;
        switch(try){
           case 0:str="1" ;break;
           case 1:str="2" ;break;
        }
        int j ;
        for( j = 0 ; j < NF ; j++ ){
             void* ptr = rec + offsets[j] ;
             int rc = sscanf( str,   fmts[j] , ptr ) ;
             printf( " j %d offset %d fmt %s rc %d \n", j, offsets[j], fmts[j], rc  );
             printf( " rc %d \n", rc );
        }
        rec += descr->elsize ;
    }

    PyObject* buf = PyBuffer_FromMemory( data , (Py_ssize_t)size) ;
    po("buf", buf ); 


    PyObject* a = PyArray_FromBuffer( buf, descr , count, (npy_intp)0 );
    po( "array", a ) ;

    return 0;
}



