#include "querydata.h"
#include <stdlib.h>
#include <stdio.h>

void* querydata( const char* sql, int* nrow, int* ncol, char* type )
{
    printf("querydata sql %s \n", sql );

    *type = 'f' ; 
    *nrow = 1 ;
    *ncol = 1 ;

    size_t size = sizeof(float)*(*nrow)*(*ncol) ;
    void* data = malloc(size);

    float* fdata = (float*)data ;
    fdata[0] = 1. ;

    return data ; 
}


