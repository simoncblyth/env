#include <cassert>
#include <iostream>
#include <string>

#include "TComplex.h"

struct TCheck
{
    double fRe ; 
    double fIm ; 

    // TComplex has a convertor like this  
    operator double () const { return fRe; }
};



int main()
{
    TComplex z(1., 1.); 
    double x = z ;    // WOW : thats a dangerous "feature" 

    std::cout << " z: " << z << std::endl ; 
    std::cout << " x: " << x << std::endl ; 


    TCheck c(1., 100.); 
    double c0 = c ; 
    std::cout << " c0: " << c0 << std::endl ; 


    return 0 ; 
}
