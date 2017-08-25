#pragma once

#include "DEMO_API_EXPORT.hh"

struct Prog ; 
struct Buf ; 

struct DEMO_API CullShader 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_InstanceTransform ;  
    static const char* vertSrc ; 
    static const char* geomSrc ; 
    static const char* fragSrc ; 

    CullShader();

    void init();
    void destroy();


    void setupTransformFilter(Buf* src_);
    void applyTransformFilter(Buf* dst);

    GLuint createTransformCullVertexArray(GLuint instanceBO) ;

    Prog* prog ;
 
    Buf*  src ; 
    Buf*  dst ; 
    GLuint culledTransformQuery ; 
    GLuint cullVertexArray ; 
    unsigned num_viz ; 

   

};


