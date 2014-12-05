// clang mkdirp.c -o mkdirp && ./mkdirp && rm mkdirp ; ls /tmp/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>


int mkdirp(const char* _path, int mode) 
{
    // directory tree creation by swapping slashes for end of string '\0'
    // then restoring the slash 
    //
    //  http://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux
    //  printf("_path %s \n", _path);

    char* path = strdup(_path);
    char* p = path ;
    int rc = 0 ;

    while (*p != '\0') 
    {
        p++;
        while(*p != '\0' && *p != '/') p++;

        char v = *p;  // hold on to the '/'
        *p = '\0';
        
        //printf("path [%s] \n", path);

        rc = mkdir(path, mode);

        if(rc != 0 && errno != EEXIST) 
        {
            *p = v;
            rc = 1;
            break ;
        }
        *p = v;
    }

    free(path); 
    return rc;
}



int main(int argc, char** argv)
{
    const int mode = 0777 ;
    const char* path = "/tmp/test/directory/creation" ;
    int rc = mkdirp(path, 0777);
    return rc ; 
}
