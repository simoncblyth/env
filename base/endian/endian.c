// cc $ENV_HOME/base/endian/endian.c -o $LOCAL_BASE/env/bin/endian && endian
/*

http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.byteorder.html

One of:

‘=’ native
‘<’ little-endian
‘>’ big-endian
‘|’ not applicable


*/

const unsigned int CHILD_BITS = 28; 
const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);


#include <stdio.h>

int main()
{
    int x = 1;

    // content of byte at the lowest memory address (of the 4 bytes) 
    char c = ((char *)&x)[0] ? '<' : '>' ;
    printf("%c",c );

    switch ( c ) 
    {
        case '<':
            printf(" : little-endian : increasing numeric significance with increasing memory addresses \n");
            break ;
        case '>':
            printf(" : big-endian    : decreasing numeric significance with increasing memory addresses \n");
            break ;
    } 
    printf("\n");
    printf("sizeof(char)            %zu \n", sizeof(char)); 
    printf("sizeof(void)            %zu \n", sizeof(void)); 
    printf("sizeof(short)           %zu \n", sizeof(short)); 
    printf("sizeof(unsigned short)  %zu \n", sizeof(unsigned short)); 
    printf("sizeof(int)             %zu \n", sizeof(int)); 
    printf("sizeof(unsigned int)    %zu \n", sizeof(unsigned int)); 
    printf("sizeof(float)           %zu \n", sizeof(float)); 
    printf("sizeof(long)            %zu \n", sizeof(long)); 
    printf("sizeof(unsigned long)   %zu \n", sizeof(unsigned long)); 
    printf("sizeof(double)          %zu \n", sizeof(double)); 
    printf("\n");
    printf("sizeof(int*)            %zu \n", sizeof(int*)); 
    printf("sizeof(float*)          %zu \n", sizeof(float*)); 
    printf("sizeof(double*)         %zu \n", sizeof(float*)); 
    printf("sizeof(char*)           %zu \n", sizeof(char*)); 
    printf("sizeof(void*)           %zu \n", sizeof(void*)); 

}


