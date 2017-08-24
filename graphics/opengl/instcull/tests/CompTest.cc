
#include <iostream>
#include "Comp.hh"
#include "Vue.hh"
#include "Cam.hh"


/*

Picking view points etc with knowlede of OpenGL frame 
to keep the matrices as simple as possible.


               Y
               |  
               |  /
               | 
               |/
               +--------X
              /
             / 
            /
           Z

*/


int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    Comp comp ; 

    float factor = 2.f ; 
    float extent = 1.f ; 

    //comp.setCenterExtent( 100, 100, 100, 10 );
    comp.setCenterExtent(  0,  0,  -1,  extent );

    Vue& v = *comp.vue ; 
    Cam& c = *comp.cam ; 

    v.setEye( 0, 0,  1)  ;   // position eye along +z 
    v.setLook(0, 0,  0)  ;   // center of region
    v.setUp(  0, 1,  0)  ; 
    c.setFocus( extent, factor );  // near/far heuristic from extent of region of interest, near = extent/factor ; far = extent*factor

      


    comp.update();

    comp.dump(); 
    comp.dumpFrustum();

   

}
