#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Prog ; 

struct DEMO_API InstRendererUniform
{  
    glm::mat4 ModelViewProjection ;
};

struct DEMO_API InstRenderer 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_VertexPosition ;  
    static const unsigned LOC_InstanceTransform ;  
    static const char* vertSrc ; 
    static const char* fragSrc ; 

    InstRenderer();

    void init();
    void initUniformBuffer();
    GLuint createVertexArray(GLuint instanceBO, GLuint vertexBO) ;
    void destroy();


    Prog*                draw ; 
    InstRendererUniform* uniform ; 
    GLuint               uniformBO ; 

    void updateMVP( const glm::mat4& w2c);
};


