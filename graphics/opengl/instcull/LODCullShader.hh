#pragma once

#include "DEMO_API_EXPORT.hh"

struct SContext ; 
struct Prog ; 
struct Buf ; 
struct Buf4 ; 

struct DEMO_API LODCullShader 
{
    static const unsigned QSIZE ;  
    static const unsigned LOC_InstanceTransform ;  
    static const char* vertSrc ; 
    static const char* geomSrc ; 

    enum { LOD_MAX = 4 } ;

    LODCullShader(SContext* context);

    void init();
    void destroy();

    // for each LOD level generate queries and bind tranform feedback stream output buffers, create single forking VAO
    void setupFork(Buf* src_, Buf4* dst_);

    void applyFork();
    void applyForkStreamQueryWorkaround();
    void pullback();
    void dump(const char* msg);

    GLuint createForkVertexArray(GLuint instanceBO) ;

    SContext* context ; 
    Prog* prog ;
 
    Buf*   src ; 
    Buf4*  dst ; 

    GLuint lodQuery[LOD_MAX] ; 
    GLuint forkVertexArray ; 

    int num_lod ;

    GLuint LOC_WORKAROUND ;
    GLint WORKAROUND ; 
 
 
};


