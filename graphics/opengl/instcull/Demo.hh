#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>


struct Transforms ; 
struct Frame ; 
struct Prog  ; 
struct Buf  ; 
struct Geom  ; 
struct Comp  ; 

struct DEMO_API Uniform
{
    glm::mat4 ModelView ; 
    glm::mat4 ModelViewProjection ;

    Uniform() 
        : 
        ModelView(1.f), 
        ModelViewProjection(1.f)
    {}  ;

};


struct DEMO_API Demo 
{
    static const unsigned QSIZE ; 

    static const unsigned LOC_InstanceTransform ;     
    static const char*    vertCullSrc ; 
    static const char*    geomCullSrc ; 
   
    static const unsigned LOC_VertexPosition ;  
    static const unsigned LOC_VizInstanceTransform ;  
    static const char*    vertDrawSrc ; 
    static const char*    fragDrawSrc ; 
 
    Uniform* uniform ; 
    Geom*    geom ; 
    Comp*    comp ; 
    Frame*   frame ;   
    Prog*    cull ; 
    Prog*    draw ; 

    unsigned cullVertexArray ;   
    unsigned drawVertexArray ;   
    unsigned allVertexArray ;   

    unsigned uniformBO ; 
    unsigned vertexBO ;   
    unsigned transformBO ;   

    unsigned culledTransformBO ;   
    unsigned culledTransformQuery ;   

    Demo();

    void init();

    void targetGeometry();
    void setupUniformBuffer();
    void updateUniform(float t);

    void loadMeshData();
    void upload(Buf* buf);
    void loadShaders();
    void createInstances();
    void renderScene(float t);
    void renderLoop();
    void errcheck(const char* msg );
    void pullback();
    void destroy();

    unsigned createVertexArray(unsigned instanceBO) ;



};

