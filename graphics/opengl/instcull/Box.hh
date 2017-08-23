#pragma once

#include "DEMO_API_EXPORT.hh"

#include <vector>
#include <glm/glm.hpp>

#include "BB.hh"

struct UV ; 
struct Buf ; 

struct DEMO_API Box
{
    unsigned level ;
    BB bb ; 

    Box(unsigned level_, float extent); 
    void init();
    glm::vec4 par_pos_model(const UV& uv ) ;

    Buf* buf();

    std::vector<glm::vec4> tri_vert ; 
};
