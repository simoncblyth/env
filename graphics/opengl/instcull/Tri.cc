#include "Tri.hh"
#include "Pos.hh"
#include "Buf.hh"
#include "BB.hh"

Tri::Tri(float sx_, float sy_, float sz_ , float cx_ , float cy_ , float cz_ )
    :
    sx(sx_), 
    sy(sy_), 
    sz(sz_), 
    cx(cx_), 
    cy(cy_), 
    cz(cz_),
    bb(NULL),
    buf(NULL)
{
    init();
}


void Tri::init() 
{
    GetVert(vert, sx,sy,sz, cx,cy,cz );
    bb = BB::FromVert(vert); 
    buf = Buf::Make(vert);
    ce = bb->get_center_extent();
} 

void Tri::GetVert(std::vector<glm::vec4>& vert, float sx, float sy, float sz, float cx, float cy, float cz )  
{
    // (-sx,-sy,sz,1) (-sx,sy,sz,1) (sx,0,sz,1)   centered on (cx,cy,cz)
    vert.push_back( { -sx + cx,   -sy + cy,  sz + cz , 1.f } );
    vert.push_back( { -sx + cx,    sy + cy,  sz + cz , 1.f } );
    vert.push_back( {  sx + cx,  0.0f + cy,  sz + cz , 1.f } );
}





