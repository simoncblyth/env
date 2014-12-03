// clang -g -L/opt/local/lib -lsqlite3 querydata.c test_querydata.c -o test_querydata && ./test_querydata && rm test_querydata

#include "querydata.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, const char** argv)
{
    const char* sql = "select tottime from test ; ";
    if(argc > 1) sql = argv[1]; 

    int fbufmax = 1000 ; 
    float* fbuf = (float*)malloc(fbufmax*sizeof(float));

    char type ;
    int nrow ;
    int ncol ; 

    int rc = querydata("SQLITE3_DATABASE", sql, &nrow, &ncol, &type, fbuf, fbufmax );
    if( rc != 0){
        fprintf(stderr, "Error %d from querydata\n", rc ); 
        exit(rc);
    } 
    printf("querydata ncol %d nrow %d type %c \n", ncol, nrow, type);

    for(int r = 0 ; r < nrow ; r++ )
    {
        for(int c = 0 ; c < ncol ; c++ )
        {
             int index = r*ncol + c ; 
             float f = *(fbuf + index) ; 
             printf("r %d c %d [%f]\n", r, c, f );          
        }
    }

    free(fbuf);


}
