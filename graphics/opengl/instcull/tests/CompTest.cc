
#include <iostream>
#include "Comp.hh"
#include "Vue.hh"
#include "Cam.hh"


int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    Comp comp ; 

    comp.setCenterExtent( 100, 100, 100, 10 );

    comp.vue->setEye(-1.3, -1.7, 0)  ; 

    comp.cam->setFocus( 10 );


    comp.update();

    comp.dump();
    

   

}
