#pragma once
#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Tra ; 
struct Prim ; 
struct Buf  ;
struct BB ;  

struct DEMO_API Geom
{  
    char     shape ; 

    unsigned num_vert ; 
    unsigned num_inst ; 
    unsigned num_viz ; 

    Tra* itra ; 
    Tra* ctra ; 

    Buf* vbuf ;  
    Buf* ebuf ;  
    BB*  vbb ; 

    Buf* ibuf  ;  
    BB*  ibb ; 

    Buf* cbuf  ;  

    glm::vec4 ce ; 


    Geom(char shape_);

    void init();
    void initGlobe();
    void initSpiral();

    void setPrim(Prim* prim);
    void setTransforms(Tra* tra);



};
 
   


