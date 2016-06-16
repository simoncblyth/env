
#include "shared.h"
#include "BDemo.hh"
#include <iostream>

int main(int argc, char** argv)
{
    std::cout << argv[0]
		      << " from vs " 
		      << std::endl ; 

    f();
   
    X* x = new X ;
    x->mX();

    BDemo bd(42) ; 
    bd.check(); 


    return 0 ; 
}

