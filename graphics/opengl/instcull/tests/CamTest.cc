#include <iostream>
#include <glm/glm.hpp>
#include "Cam.hh"

int main()
{
    Cam cam ; 
    std::cout << cam.desc() << std::endl ;    

    cam.zoom = 0.1 ;
    std::cout << cam.desc() << std::endl ;    


    return 0 ; 
}
