#pragma once

#include "DEMO_API_EXPORT.hh"

struct Geom  ; 
struct Comp  ; 
struct Frame ; 
struct CullShader  ; 
struct InstShader  ; 

struct DEMO_API ICDemo 
{
    static const unsigned QSIZE ; 

    Geom*        geom ; 
    Comp*        comp ; 
    Frame*       frame ;   
    CullShader*  cull ; 
    InstShader*  draw ; 

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

