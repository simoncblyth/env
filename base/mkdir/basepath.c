// clang basepath.c -o basepath && ./basepath && rm basepath 


#include <string.h>
#include <stdio.h>


char* basepath( const char* _path, char delim )
{
    char* path = strdup(_path);
    char* dot  = strrchr(path, delim) ;  // returns NULL when delim not found
    if(dot) *dot = '\0' ;
    return path ; 
}


int main()
{
    const char* _path = "/usr/local/env/tmp/20140514-174932.npy";

    printf("_path %s \n", _path);
    printf("path %s \n", basepath(_path,'.'));
    printf("path %s \n", basepath(_path,'/'));

    return 0 ;
}
