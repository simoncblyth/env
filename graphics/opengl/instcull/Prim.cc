#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>


#include "Prim.hh"
#include "Buf.hh"
#include "BB.hh"
#include "G.hh"


Prim::Prim()
    :
    vbuf(NULL),
    ebuf(NULL),
    bb(NULL)
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

    unsigned eOffset = 0u ;
    unsigned num_elem = ebuf->num_items ;
    unsigned vOffset = 0u ;
    unsigned num_vert = vbuf->num_items ;

    eidx.push_back( {  eOffset, num_elem, vOffset, num_vert } );
}


Prim* Prim::Concatenate( std::vector<Prim*> prims )
{
    uint32_t ebufSize = 0;
    uint32_t vbufSize = 0;

    for(uint32_t p=0 ; p < prims.size() ; p++) 
    {
        Prim* prim = prims[p];
        ebufSize += prim->ebuf->num_items ; 
        vbufSize += prim->vbuf->num_items ; 
    }
       
    uint32_t* edat =  new uint32_t[ebufSize] ;
    glm::vec4* vdat = new glm::vec4[vbufSize];

    Prim* concat = new Prim ; 

    std::vector<glm::uvec4>& eidx = concat->eidx ; 
    concat->ebuf = new Buf( ebufSize , sizeof(uint32_t)*ebufSize , edat );
    concat->vbuf = new Buf( vbufSize , sizeof(glm::vec4)*vbufSize , vdat );

    unsigned eOffset = 0;
    unsigned vOffset = 0;

    for(uint32_t p=0 ; p < prims.size() ; p++) 
    {
        Prim* prim = prims[p];
        uint32_t num_elem = prim->ebuf->num_items ;
        uint32_t num_vert = prim->vbuf->num_items ;

        for (uint32_t e=0; e < num_elem ; e++) edat[eOffset+e] = *((uint32_t*)prim->ebuf->ptr + e) + vOffset ;

        eidx.push_back( {  eOffset, num_elem, vOffset, num_vert } );
        
        memcpy( (void*)( vdat + vOffset ), prim->vbuf->ptr , prim->vbuf->num_bytes );
        eOffset += num_elem ;
        vOffset += num_vert ;
    }

    concat->bb = BB::FromBuf(concat->vbuf);
    concat->ce = concat->bb->get_center_extent();
 
    return concat ;     
}


void Prim::dump(const char* msg)
{
    std::cout << msg << std::endl ; 
    std::cout << bb->desc() << std::endl ; 
    std::cout << G::gpresent("ce", ce) << std::endl ; 

    for(unsigned i=0 ; i < eidx.size() ; i++)
        std::cout << G::gpresent("eidx", eidx[i]) << std::endl ; 

    ebuf->dump("ebuf");
    vbuf->dump("vbuf");

}



