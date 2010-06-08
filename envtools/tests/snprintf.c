#include <stdio.h>



int dump( char* buf , size_t bufsize , char* p )
{
    printf(" p - buf     : %d \n" , p - buf );
    printf(" buf : %s \n", buf );
}


int collect_colors(char* buf, size_t bufsize )
{
    char* p = buf ;
    printf( "bufsize     : %d \n" , bufsize );
    
    
    char* r = "red" ;
    p += snprintf( p,  bufsize - (p - buf),  "%s ", r );
    dump( buf , bufsize , p );
    
    char* g = "green" ;
    p += snprintf( p,  bufsize - (p - buf),  "%s ", g );
    dump( buf , bufsize , p );
    
    char* b = "blue" ;
    p += snprintf( p,  bufsize - (p - buf),  "%s ", b );
    dump( buf , bufsize , p );
    
    if( bufsize < p - buf ){
        printf("bufsize %d is too small ... ", bufsize) ;
        buf[0] = 0 ;
    }
}


int main()
{
    char cols[20] ;
    collect_colors( cols , sizeof(cols) );
    printf( "collected ... %s\n", cols );
    return 0 ;
}