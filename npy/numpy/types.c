/*
*/
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#include "mysql_numpy.h"


int main(int argc, char *argv[])
{
    Py_Initialize();
    import_array();


    int i ;
    for( i=0 ; i<NPY_NTYPES ; ++i ){

       printf(" %2d %-20s %5s \n", i ,  NPY_TYPE_NAMES[i], NPY_TYPE_FMTS[i]  );

    } 

 


    return 0 ;
}
