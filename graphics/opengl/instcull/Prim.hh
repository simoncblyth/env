#pragma once

#include <stdint.h>
#include <vector>
#include <glm/glm.hpp>

#include "DEMO_API_EXPORT.hh"

struct Buf ; 
struct BB ; 

struct DEMO_API Prim 
{
    static Prim* Concatenate( std::vector<Prim*> prims );

    Buf* vbuf ; 
    Buf* ebuf ; 
    std::vector<glm::uvec4> eidx ; 


    // derived
    BB* bb ; 
    glm::vec4 ce ; 

    // temporary inputs 
    std::vector<glm::vec4>  vert ; 
    std::vector<unsigned>   elem ; 

    Prim();

    unsigned add_vert(float x, float y ,float z);
    unsigned add_vert(const glm::vec3& p);

    void add_tri(unsigned v0, unsigned v1, unsigned v2);
    void add_quad(unsigned v0, unsigned v1, unsigned v2, unsigned v3);

    void populate();  // call after vert and elem are filled

    void dump(const char* msg);


};

