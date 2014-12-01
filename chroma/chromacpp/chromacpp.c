/* 

    clang npyreader.c -c     
    clang chromacpp.c -c     
    clang *.o -o chromacpp

*/

#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>

#include "npyreader.h"


int readnpy(const char* path, const char* root )
{
    FILE *fp;

    fp = fopen(path, "r");

    if (fp == NULL){
        printf("COULD NOT OPEN FILE: %s", path);
        return -1; 
    }   

    char* dict = retrieve_python_dict(fp);
    char* dtype = get_descr( dict );
    char* shape = get_shapestr( dict );

    uint32_t* fsizes = get_shape_file(fp);

    size_t i = 0 ;
    size_t datasize = 1  ;  
    while(fsizes[i] != -1){
        datasize *= fsizes[i] ;
        i++;
    }   

    size_t rlen = strlen(root) ;
    size_t plen = strlen(path) ; 
    assert( plen > rlen );

    printf("\t NPY dt %5s sh %10s sz %10zu  %s \n", dtype, shape, datasize, path + rlen + 1 );

    free(fsizes);
    free(dtype);
    free(shape);

    if( strcmp(dtype,"<f4") == 0 )
    {
        float32_t* data = retrieve_npy_float32(fp);
        if (data == NULL) return 6;
        //for(size_t i=0 ; i < datasize ; ++i ) if( i < 5 || i > datasize-5 ) printf("\t %zu    %f   \n", i, data[i] ); 
        free(data);
    }
    else if( strcmp(dtype,"<i4") == 0 )
    {
        int32_t* data = retrieve_npy_int32(fp);
        if (data == NULL) return 6;
        //for(size_t i=0 ; i < datasize ; ++i ) if( i < 5 || i > datasize-5 ) printf("\t %zu    %d   \n", i, data[i] ); 
        free(data);
    }
    else if( strcmp(dtype,"<u4") == 0 )
    {

    }


    fclose(fp);
    return 0;
}



int recursive_listdir( const char* base, const char* root )
{
    size_t rlen = strlen(root) ;
    size_t blen = strlen(base) ; 
    assert( blen >= rlen );

    DIR* dir ; 
    struct dirent* dent;
    struct stat st;
    char path[1024];

    if (!(dir = opendir(base)))
        return -1;

    while ((dent = readdir(dir)) != NULL) 
    {
        int len = snprintf(path, sizeof(path)-1, "%s/%s", base, dent->d_name);
        path[len] = 0;

        lstat(path, &st);

        if(S_ISDIR(st.st_mode))
        {
            if(strcmp(dent->d_name,".") == 0 || strcmp(dent->d_name,"..") == 0 ) continue ;
            else
            {
                printf("\t D %s \n", path + rlen + 1 );
                recursive_listdir( path, root );  
            } 
        }
        else
        {
            int flen = strlen(dent->d_name);
            if( flen < 4 ){
                printf("\t F SKIP short filename  %s \n", path );
                continue ; 
            } 

            const char* flast = dent->d_name + flen - 4 ;
            if( strcmp( flast, ".npy") == 0 ){
                  readnpy( path, root );
            } 
            else 
            {
                  printf("\t OTHER %s \n", path + rlen + 1 );
            }

        }
    } 
    closedir(dir); 
    return 0;
}


int main(int argc, char **argv )
{
    if(argc < 2){ 
        printf("Expecting directory string argument\n");
        return -1;  
    }

    const char* root = argv[1] ;

    int rc = recursive_listdir(root, root); 

    return rc ;
}
