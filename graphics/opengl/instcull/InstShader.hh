#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct Prog ; 
struct SContext ; 

struct DEMO_API InstShader 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_VertexPosition ;  
    static const unsigned LOC_VizInstanceTransform ;  
    static const char*    vertSrc ; 
    static const char*    fragSrc ; 

    InstShader(SContext* context);

    void initProgram();
    //void initUniformBuffer();
    GLuint createVertexArray(GLuint instanceBO, GLuint vertexBO) ;
    void destroy();


    SContext*            context ; 
    Prog*                prog ; 


};


