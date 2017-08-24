#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "DEMO_API_EXPORT.hh"

struct Buf ; 
struct BB ; 

struct DEMO_API Prim 
{
    BB* bb ; 
    Buf* vbuf ; 
    Buf* ebuf ; 
    glm::vec4 ce ; 
    std::vector<glm::vec4> vert ; 
    std::vector<unsigned>  elem ; 

    Prim();
    void add_tri(unsigned v0, unsigned v1, unsigned v2);
    void add_quad(unsigned v0, unsigned v1, unsigned v2, unsigned v3);

    void populate();  // call after vert and elem are filled

};

