#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Prog ; 

struct DEMO_API InstShaderUniform
{  
    glm::mat4 ModelViewProjection ;
};

struct DEMO_API InstShader 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_VertexPosition ;  
    static const unsigned LOC_VizInstanceTransform ;  
    static const char*    vertSrc ; 
    static const char*    fragSrc ; 

    InstShader();

    void initProgram();
    void initUniformBuffer();
    GLuint createVertexArray(GLuint instanceBO, GLuint vertexBO) ;
    void destroy();


    Prog*                prog ; 
    InstShaderUniform*   uniform ; 
    GLuint               uniformBO ; 

    void updateMVP( const glm::mat4& w2c);
};


