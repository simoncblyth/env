/*
  On OSX ...   THE %Ld FORMAT IS NOT APPRECIATED 

simon:numpy blyth$ make run
./datetime_buf
op
[('a_int', 'i4'), ('a_float', 'f4'), ('a_datetime', 'M8[s]'), ('b_datetime', 'M8[s]')]
descr
dtype([('a_int', '>i4'), ('a_float', '>f4'), ('a_datetime', ('>M8[s]', {})), ('b_datetime', ('>M8[s]', {}))])
 j 0 offset 0 fmt %d rc 1 
 rc 1 
 j 1 offset 4 fmt %g rc 1 
 rc 1 
 j 2 offset 8 fmt %lld rc 1 
 rc 1 
 j 3 offset 16 fmt %Ld rc 1 
 rc 1 
 j 0 offset 0 fmt %d rc 1 
 rc 1 
 j 1 offset 4 fmt %g rc 1 
 rc 1 
 j 2 offset 8 fmt %lld rc 1 
 rc 1 
 j 3 offset 16 fmt %Ld rc 1 
 rc 1 
buf
<read-only buffer ptr 0x470e70, size 48 at 0x1115c20>
array
array([ (1, 1.0, datetime.datetime(1970, 1, 1, 0, 0, 1), datetime.datetime(2106, 2, 7, 6, 28, 16)),
       (2, 2.0, datetime.datetime(1970, 1, 1, 0, 0, 2), datetime.datetime(2242, 3, 16, 12, 56, 35))], 
      dtype=[('a_int', '>i4'), ('a_float', '>f4'), ('a_datetime', ('>M8[s]', {})), ('b_datetime', ('>M8[s]', {}))])



  On Linux ...    THE %Ld WORKS  (as does %lld ... which works on OSX)

[blyth@cms01 numpy]$ make run
./datetime_buf
op
[('a_int', 'i4'), ('a_float', 'f4'), ('a_datetime', 'M8[s]'), ('b_datetime', 'M8[s]')]
descr
dtype([('a_int', '<i4'), ('a_float', '<f4'), ('a_datetime', ('<M8[s]', {})), ('b_datetime', ('<M8[s]', {}))])
 j 0 offset 0 fmt %d rc 1 
 rc 1 
 j 1 offset 4 fmt %g rc 1 
 rc 1 
 j 2 offset 8 fmt %lld rc 1 
 rc 1 
 j 3 offset 16 fmt %Ld rc 1 
 rc 1 
 j 0 offset 0 fmt %d rc 1 
 rc 1 
 j 1 offset 4 fmt %g rc 1 
 rc 1 
 j 2 offset 8 fmt %lld rc 1 
 rc 1 
 j 3 offset 16 fmt %Ld rc 1 
 rc 1 
buf
<read-only buffer ptr 0x92592b0, size 48 at 0xb7b73cc0>
array
array([ (1, 1.0, datetime.datetime(1970, 1, 1, 0, 0, 1), datetime.datetime(1970, 1, 1, 0, 0, 1)),
       (2, 2.0, datetime.datetime(1970, 1, 1, 0, 0, 2), datetime.datetime(1970, 1, 1, 0, 0, 2))], 
      dtype=[('a_int', '<i4'), ('a_float', '<f4'), ('a_datetime', ('<M8[s]', {})), ('b_datetime', ('<M8[s]', {}))])
[blyth@cms01 numpy]$ 









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
     po("descr", (PyObject*)descr );

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



