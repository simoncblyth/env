#include <iostream>

#include "Buf.hh"
#include "BB.hh"
#include "Tra.hh"
#include "Geom.hh"

#include <glm/gtc/type_ptr.hpp>

const unsigned Geom::QSIZE = sizeof(float)*4 ; 


struct V { float x,y,z,w ; };
struct V4 { V x,y,z,w ; };



Geom::Geom(unsigned num_vert_, unsigned num_inst_)
    :
    num_vert(num_vert_),
    num_inst(num_inst_),
    num_viz(0),

    itra(new Tra(num_inst, 4, 4)),
    ctra(new Tra(num_inst, 4, 4)),

    vbuf(new Buf(num_vert, QSIZE*1*num_vert,   NULL )),  
    vbb(new BB),

    ibuf(new Buf(num_inst, QSIZE*4*num_inst, itra->data )),  
    ibb(new BB),

    cbuf(new Buf(num_inst, QSIZE*4*num_inst, ctra->data ))
{
    init();
}

void Geom::init()
{
    if( num_vert == 3 )
    {
        vbuf->ptr = new V[num_vert] ; 

        V* vptr = (V*)vbuf->ptr ; 

        vptr[0] = {  0.05f ,   0.05f,  0.00f,  1.f } ;
        vptr[1] = {  0.05f ,  -0.05f,  0.00f,  1.f } ;
        vptr[2] = {  0.00f ,   0.00f,  0.00f,  1.f } ;

        update_bounds();
    }


    
}


void Geom::update_bounds()
{
    for(unsigned i=0 ; i < num_vert ; i++)
    {
         V* vptr = (V*)vbuf->ptr ; 
         glm::vec3 p = glm::make_vec3( (float*)(vptr + i) );
         vbb->include(p);
    }


    for(unsigned i=0 ; i < num_inst ; i++)
    {
         V4* iptr = ((V4*)ibuf->ptr) + i ; 
         V*  vptr = ((V*)iptr + 3 );

         glm::vec3 p = glm::make_vec3( (float*)(vptr) );
         ibb->include(p);
    }

    std::cout << "Geom::update_bounds "
              << std::endl 
              << " vbb " << vbb->desc()
              << std::endl 
              << " ibb " << ibb->desc()
              << std::endl 
              ;

}











