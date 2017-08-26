#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Prog ; 

struct DEMO_API ShaderUniform
{  
    glm::mat4 ModelViewProjection ;
};

struct DEMO_API Shader 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_VertexPosition ;  
    static const char*    vertSrc ; 
    static const char*    fragSrc ; 

    Shader();

    void init();
    void initUniformBuffer();

    GLuint createVertexArray(GLuint vertexBO, GLuint elementBO) ;
    void destroy();

    Prog*                draw ; 
    ShaderUniform*     uniform ; 
    GLuint               uniformBO ; 

    void updateMVP( const glm::mat4& w2c);
};


