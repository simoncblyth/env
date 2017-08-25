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

    void setupFork(Buf* src_, Buf4* dst_);
    void applyFork();

    GLuint createForkVertexArray(GLuint instanceBO) ;

    SContext* context ; 
    Prog* prog ;
 
    Buf*   src ; 
    Buf4*  dst ; 

    GLuint lodQuery[LOD_MAX] ; 
    int    lodCount[LOD_MAX] ; 

    GLuint forkVertexArray ; 

    int num_lod ; 
    int num_viz ;    

};


