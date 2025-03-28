// clang numutil.cpp -o /tmp/numutil
#include "limits.h"
#include "stdio.h"
#include "assert.h"


#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
// http://stackoverflow.com/questions/7337526/how-to-tell-if-a-32-bit-int-can-fit-in-a-16-bit-short


bool fits_short_0(int x )
{
    return int(short(x)) == x ;   // from the horses mouth 
}

bool fits_short_1(int x )
{
    return ((x & 0xffff8000) + 0x8000) & 0xffff7fff;  // INCORRECT ???
}

bool fits_short_2(int x)
{
    return fitsInShort(x);
}

#define N 3
int main()
{

    int count[N] ; 
    for(char i=0 ; i < N ; i++) count[i] = 0 ;

    //for(int i=INT_MIN ; i < INT_MAX ; i++)
    for(int i=SHRT_MIN*10 ; i < SHRT_MAX*10 ; i++)
    {
        bool b0 = fits_short_0(i) ;
        bool b1 = fits_short_1(i) ;
        bool b2 = fits_short_2(i) ;

        if(b0) count[0] += 1 ; 
        if(b1) count[1] += 1 ; 
        if(b2) count[2] += 1 ; 

        assert( b0 == b2 );

    }

    for(char i=0 ; i < N ; i++)
    {
        printf("count %d : %d \n", i, count[i]);
    }
    return 0 ; 
}

