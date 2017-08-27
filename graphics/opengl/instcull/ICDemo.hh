#pragma once

#define WITH_LOD 1

#include "DEMO_API_EXPORT.hh"
#include <string>

struct Geom  ; 
struct Comp  ; 
struct Frame ; 
struct SContext  ; 

#ifdef WITH_LOD
struct Buf4  ; 
struct LODCullShader  ; 
#else
struct CullShader  ; 
#endif

struct InstShader  ; 

struct DEMO_API ICDemo 
{
    static const unsigned QSIZE ; 

    Geom*        geom ; 
    Comp*        comp ; 
    Frame*       frame ;   
    SContext*    context ; 

#ifdef WITH_LOD
    LODCullShader*  cull ; 
    Buf4*           clod ; 
    unsigned        num_lod ; 
#else
    CullShader*  cull ; 
#endif

    InstShader*  draw ; 
    bool         use_cull ; 

    unsigned drawVertexArray[4] ;   
    unsigned allVertexArray ;   

    ICDemo(const char* title);

    void init();
    void updateUniform(float t);
    void renderScene();
    void renderLoop();
    void pullback();
    void destroy();

    std::string getStatus();

};

