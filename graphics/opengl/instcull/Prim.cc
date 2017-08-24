#include "Prim.hh"
#include "Buf.hh"
#include "BB.hh"


Prim::Prim()
    :
    bb(NULL),
    vbuf(NULL),
    ebuf(NULL)
{
}

unsigned Prim::add_vert(float x, float y, float z)
{
    unsigned num_vert = vert.size();
    vert.push_back( { x, y , z, 1.f } );
    return num_vert ; 
}

unsigned Prim::add_vert(const glm::vec3& p)
{
    return add_vert(p.x, p.y, p.z);
}

void Prim::add_tri(unsigned v0, unsigned v1, unsigned v2)
{
    elem.push_back(v0);
    elem.push_back(v1);
    elem.push_back(v2);
}

void Prim::add_quad(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
{
    /*  
              3-------2
              |     . | 
              |   .   |
              | .     |
              0-------1  
   */
    add_tri(v0,v1,v2);
    add_tri(v2,v3,v0);
}

void Prim::populate()
{
    bb = BB::FromVert(vert); 
    vbuf = Buf::Make(vert);
    ebuf = Buf::Make(elem);
    ce = bb->get_center_extent();
}




