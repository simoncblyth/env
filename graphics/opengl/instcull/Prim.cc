
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



Prim* Prim::Concatenate( std::vector<Prim*> prims )
{
    uint32_t ebufSize = 0;
    for(uint32_t i=0 ; i < prims.size() ; i++) 
        ebufSize += prims[i]->ebuf->num_items ; 
       
    uint32_t* edat =  new uint32_t[ebufSize] ;
    Buf* ebuf = new Buf( ebufSize , sizeof(uint32_t)*ebufSize , edat );

    unsigned eOffset = 0;
    unsigned vOffset = 0;
    for(uint32_t i=0 ; i < prims.size() ; i++) 
        Concatenate(edat, eOffset, vOffset, prims[i]);


    uint32_t vbufSize = 0;
    for(uint32_t i=0 ; i < prims.size() ; i++) 
        vbufSize += prims[i]->vbuf->num_items ; 
 
    PV* vdat = new PV[vbufSize]; 

    return NULL ;     
}

void Prim::Concatenate(uint32_t* ptr, uint32_t& eOffset, uint32_t& vOffset, Prim* prim ) 
{
    for (uint32_t i=0; i < prim->ebuf->num_items ; i++)
        ptr[eOffset+i] = *((uint32_t*)prim->ebuf->ptr) + vOffset ;

    eOffset += prim->ebuf->num_items ;
    vOffset += prim->vbuf->num_items ;
}


void Prim::Concatenate(PV* ptr, uint32_t& eOffset, uint32_t& vOffset, Prim* prim ) 
{
    //for (uint32_t i=0; i < prim->vbuf->num_items ; i++)
    //{
    //    ptr[eOffset+i].x = ((V*)prim->vbuf->ptr) + vOffset ;
    //} 
    //
    //eOffset += prim->ebuf->num_items ;
    //vOffset += prim->vbuf->num_items ;
}


















