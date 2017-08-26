#include <vector>
#include <cassert>

#include "Cube.hh"
#include "Sphere.hh"

int main()
{
    Cube*   cube = new Cube ;
    Sphere* sphere = new Sphere(2u,5.f) ;

    cube->dump("cube");
    sphere->dump("sphere");



    std::vector<Prim*> prims ; 
    prims.push_back(cube);
    prims.push_back(sphere);


    Prim* prim = Prim::Concatenate(prims);
    prim->dump("Prim::Concatenate(cube, sphere)");
    


    return 0 ;  
}

