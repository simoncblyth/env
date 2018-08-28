
// https://en.wikipedia.org/wiki/Middle-square_method

#include <stdio.h>
#include <stdint.h>

uint64_t x = 0, w = 0, s = 0xb5ad4eceda1ce2a9;

inline static uint32_t msws() {

   x *= x; 
   x += (w += s); 
   return x = (x>>32) | (x<<32);

}


int main(int argc, char** argv)
{
    for(int i=0 ; i < 10 ; i++)
    {
        uint32_t u = msws() ; 
        printf( "%d \n", u );  
    }

    return 0 ; 
}

