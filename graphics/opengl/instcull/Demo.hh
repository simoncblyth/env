#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>


struct Transforms ; 
struct Frame ; 
struct Prog  ; 
struct Buf  ; 
struct Geom  ; 
struct Comp  ; 
struct BB  ; 

struct DEMO_API Uniform
{
    glm::mat4 ModelView ; 
    glm::mat4 ModelViewProjection ;
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
    unsigned elementBO ;   
    unsigned transformBO ;   

    unsigned culledTransformBO ;   
    unsigned culledTransformQuery ;   

    Demo(const char* title);

    void init();

    //void targetGeometry(BB* bb);
    void setupUniformBuffer();
    void updateUniform(float t);

    void loadMeshData(Geom* geom);
    void upload(Buf* buf);
    void loadShaders();
    void createInstances(Buf* ibuf);
    void renderScene(float t);
    void renderLoop();
    void pullback();
    void destroy();

    unsigned createTransformCullVertexArray(unsigned instanceBO, unsigned loc) ;
    unsigned createInstancedRenderVertexArray(unsigned instanceBO) ;

};

