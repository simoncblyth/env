#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "DEMO_API_EXPORT.hh"
struct Buf ; 
struct BB ; 

struct DEMO_API Tri
{
    float sx ; 
    float sy ; 
    float sz ; 
    float cx ; 
    float cy ; 
    float cz ; 

    BB* bb ; 
    Buf* buf ; 
    glm::vec4 ce ; 
    std::vector<glm::vec4> vert ; 


    Tri( float sx_, float sy_, float sz_ , float cx_ , float cy_ , float cz );
    void init();
    static void GetVert(std::vector<glm::vec4>& vert, float sx, float sy, float sz, float cx, float cy, float cz) ;


};



