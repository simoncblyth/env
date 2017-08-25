#pragma once

#include "DEMO_API_EXPORT.hh"
#include <vector>
#include <glm/glm.hpp>

struct Buf ; 
struct BB ; 
struct UV ; 

struct DEMO_API Tra
{
    static const unsigned QSIZE ;

    unsigned ni ; 
    unsigned nj ; 
    unsigned nk ; 

    Buf*      buf ; 
    BB*       bb ; 
    glm::vec4 ce ; 

    static Tra* MakeGlobe(float radius, unsigned nu, unsigned nv ) ;
    static glm::vec3 SpherePos(const UV& uv, float radius);


    Tra(unsigned ni_, char shape_) ;
    Tra(const std::vector<glm::mat4>& mat) ;

    void populate(const std::vector<glm::mat4>& mat) ;
    void mockup(std::vector<glm::mat4>& mat, char shape);
    void mockup_spiral( glm::mat4& m , float fr );
    void mockup_diagonal( glm::mat4& m , float fr );


    void dump(unsigned n=0);

    unsigned num_items() const ;
    unsigned num_bytes() const ;
    unsigned num_floats() const ;

};

