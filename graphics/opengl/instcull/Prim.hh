#pragma once

#include <stdint.h>
#include <vector>
#include <glm/glm.hpp>

#include "DEMO_API_EXPORT.hh"

struct Buf ; 
struct BB ; 


struct DEMO_API PV { float x, y, z, w ; } ;

struct DEMO_API Prim 
{
    static Prim* Concatenate( std::vector<Prim*> prims );
    static void Concatenate(uint32_t* ptr, uint32_t& eOffset, uint32_t& vOffset, Prim* prim ) ;
    static void Concatenate(PV*       ptr, uint32_t& eOffset, uint32_t& vOffset, Prim* prim ) ;

    BB* bb ; 
    Buf* vbuf ; 
    Buf* ebuf ; 
    glm::vec4 ce ; 
    std::vector<glm::vec4> vert ; 
    std::vector<unsigned>  elem ; 

    Prim();

    unsigned add_vert(float x, float y ,float z);
    unsigned add_vert(const glm::vec3& p);

    void add_tri(unsigned v0, unsigned v1, unsigned v2);
    void add_quad(unsigned v0, unsigned v1, unsigned v2, unsigned v3);

    void populate();  // call after vert and elem are filled

};

