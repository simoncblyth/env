#include "Tri.hh"
#include "Pos.hh"
#include "Buf.hh"
#include "BB.hh"

Tri::Tri(float sx_, float sy_, float sz_ , float cx_ , float cy_ , float cz_ )
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

void Tri::init() 
{
    // (-sx,-sy,sz,1) (-sx,sy,sz,1) (sx,0,sz,1)   centered on (cx,cy,cz)

    vert.push_back( { -sx + cx,   -sy + cy,  sz + cz , 1.f } );
    vert.push_back( { -sx + cx,    sy + cy,  sz + cz , 1.f } );
    vert.push_back( {  sx + cx,  0.0f + cy,  sz + cz , 1.f } );

    elem.push_back(0); 
    elem.push_back(1); 
    elem.push_back(2); 

    populate();
}





