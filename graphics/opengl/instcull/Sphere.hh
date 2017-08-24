#pragma once

#include "DEMO_API_EXPORT.hh"
#include "Prim.hh"
#include <glm/glm.hpp>

struct UV ; 

struct DEMO_API Sphere  : Prim
{
    float radius ; 
    glm::vec3 center ; 
    int level ; 

    Sphere(float r=1.f , float cx=0.f, float cy=0.f, float cz=0.f, int level_=4 );
    void init();

    glm::vec3 par_pos_model(const UV& uv) const ;
    static void _par_pos_body(glm::vec3& pos,  const UV& uv, const float r_ ) ; 

};





