#include <iostream>
#include "Vue.hh"

int main()
{
    Vue v ; 

    v.setLook(0,0,0);
    v.setEye(0,0,1);
    v.setUp(0,1,0);

    std::cout << v.desc() << std::endl ; 
    v.setEye(0,0,10);
    std::cout << v.desc() << std::endl ; 
    


    return 0 ; 
}
