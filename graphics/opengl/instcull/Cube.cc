#include "Cube.hh"
#include "Pos.hh"
#include "Buf.hh"
#include "BB.hh"

Cube::Cube(float sx_, float sy_, float sz_ , float cx_ , float cy_ , float cz_ )
    :
    Prim(),
    sx(sx_), 
    sy(sy_), 
    sz(sz_), 
    cx(cx_), 
    cy(cy_), 
    cz(cz_)
{
    init();
}

void Cube::init() 
{
/*
    Z-order indices

                110---------111
                /|          /| 
               / |         / |
             010---------011 | 
              |  |        |  |
              |  |        |  |
              | 100-------|-101
              | /         | /
              |/          |/
             000---------001     --> X
             ZYX


                 6-----------7
                /|          /| 
               / |         / |
              2-----------3  | 
              |  |        |  |
              |  |        |  |
              |  4--------|--5
              | /         | /
              |/          |/
              0-----------1     --> X
 
*/    

    for(unsigned i=0 ; i < 8 ; i++) vert.push_back( 
            {
              i & 1 ? cx + sx : cx - sx , 
              i & 2 ? cy + sy : cy - sy , 
              i & 4 ? cz + sz : cz - sz ,
              1.f
            } );



   
    add_quad( 0, 1, 3, 2);  // front    NB consistent winding
    add_quad( 5, 4, 6, 7);  // back 
    add_quad( 1, 5, 7, 3);  // right 
    add_quad( 4, 0, 2, 6);  // left
    add_quad( 2, 3, 7, 6);  // top
    add_quad( 4, 5, 1, 0);  // bottom

    populate();
}





