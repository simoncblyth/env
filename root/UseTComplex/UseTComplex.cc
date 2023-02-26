#include <cassert>
#include <iostream>
#include <string>

#include "TComplex.h"

int main()
{
    TComplex z(1., 1.); 
    double x = z ;    // WOW : thats a dangerous "feature" 

    std::cout << " z: " << z << std::endl ; 
    std::cout << " x: " << x << std::endl ; 

    return 0 ; 
}
