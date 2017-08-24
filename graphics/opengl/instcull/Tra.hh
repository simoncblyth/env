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

    std::vector<glm::mat4> mat ; 
    Buf*      buf ; 
    BB*       bb ; 
    glm::vec4 ce ; 


    Tra(unsigned ni_, char shape_) ;

    void mockup(char shape_);
    void mockup_spiral( glm::mat4& m , float fr );
    void mockup_diagonal( glm::mat4& m , float fr );

    void mockup_globe( unsigned nu, unsigned nv, float radius);
    glm::vec3 sphere_pos(const UV& uv, float radius);


    void dump(unsigned n=0);

    unsigned num_items() const ;
    unsigned num_bytes() const ;
    unsigned num_floats() const ;

};

