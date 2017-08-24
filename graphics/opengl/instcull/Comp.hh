
/*

world frame
    asis 3D coordinate space 

model frame
    uniform extent scaled and translated frame putting 
    an object of arbitrary ce (center extent) in the world frame
    within  -1:1 cube of the model frame

    NB does not preclude going outside the unit cube, it just 
    makes the meaning of doing so the same no matter the size 
    of the world frame object 


*/

#pragma once

#include "DEMO_API_EXPORT.hh"

#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Vue ; 
struct Cam ; 

struct DEMO_API Comp
{
    static std::string gpresent(const char* label, const glm::mat4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=20, unsigned mwid=5, bool flip=false );
    static std::string gpresent(const char* label, const glm::vec4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=20, unsigned mwid=5);

    // inputs from setCenterExtent picking a world frame region of interest 
    glm::vec4 center_extent ;   
    glm::mat4 model2world ;   
    glm::mat4 world2model ; 

    // from camera
    glm::mat4 projection ; 

    // from View::getTransforms
    glm::mat4 world2camera ; 
    glm::mat4 camera2world ; 
    glm::vec4 gaze ; 
    
    // derived from above
    glm::mat4 world2clip ;   // ModelViewProjection
    glm::mat4 world2eye ;    // ModelView 

    Vue* vue ; 
    Cam* cam ; 

    Comp();

    void setCenterExtent(float x, float y, float z, float w);
    void setCenterExtent(const glm::vec4& ce);  
    void update();
    void dump();
    void dumpCorners();
    void dumpFrustum();
    void dumpPoints(const std::vector<glm::vec4>& world);

    glm::vec3 getNDC(const glm::vec4& world);
    glm::vec3 getNDC2(const glm::vec4& world);

    std::string desc();
};





