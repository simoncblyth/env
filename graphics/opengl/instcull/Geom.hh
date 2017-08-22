#pragma once
#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Tra ; 
struct Buf  ;
struct BB ;  

struct DEMO_API Geom
{  
    static const unsigned QSIZE ; 
    unsigned num_vert ; 
    unsigned num_inst ; 
    unsigned num_viz ; 

    Tra* itra ; 
    Tra* ctra ; 

    Buf* vbuf ;  
    BB*  vbb ; 

    Buf* ibuf  ;  
    BB*  ibb ; 


    Buf* cbuf  ;  

    Geom(unsigned num_vert, unsigned num_inst);
    void init();
    void update_bounds();

};
 
   


