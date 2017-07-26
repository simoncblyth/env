#include <stdio.h>
#include "Roots3And4.c"




void test_SolveQuadric_1()
{
    double c[3] ; 
    double s[2] ; 

    //  (x - 1)*(x - 1) = x^2 -2x + 1 
    c[2] =  1. ; 
    c[1] = -2. ; 
    c[0] =  1. ; 
  
    int n = SolveQuadric( c, s);

    printf(" n %d \n", n );

    for(int i=0 ; i < n ; i++) printf(" s[%d] = %f \n", i, s[i] ) ;
}




int main(int argc, char** argv)
{

    test_SolveQuadric_1();

    return 0 ; 
}
