#pragma once

#include "DEMO_API_EXPORT.hh"

struct Geom  ; 
struct Comp  ; 
struct Frame ; 
struct SContext  ; 
struct CullShader  ; 
struct InstShader  ; 

struct DEMO_API ICDemo 
{
    static const unsigned QSIZE ; 

    Geom*        geom ; 
    Comp*        comp ; 
    Frame*       frame ;   
    SContext*    context ; 
    CullShader*  cull ; 
    InstShader*  draw ; 
    bool         use_cull ; 

    unsigned drawVertexArray ;   
    unsigned allVertexArray ;   

    ICDemo(const char* title);

    void init();
    void updateUniform(float t);
    void renderScene(float t);
    void renderLoop();
    void pullback();
    void destroy();

};

