#pragma once

#include "DEMO_API_EXPORT.hh"
#include <string>
#include <vector>

struct DEMO_API Cam
{
    static const char* doc ; 
    
    int   width ;  
    int   height ;  
    float basis ; 
    float factor ; 
    float near ; 
    float far ; 
    float zoom ; 

    bool parallel ; 
    float parascale ; 

    Cam( int width_=1024, int height_=768, float basis=1000.f );

    void setSize(int width, int height );
    void setFocus(float basis, float factor=10.);
    void setYfov(float yfov_deg);

    float getAspect() const ; 
    float getScale() const ; 
    float getTop() const ;
    float getBottom() const ;
    float getLeft() const ;
    float getRight() const ;
    float getNear() const ;
    float getFar() const ;
    float getYfov() const ;

    glm::mat4 getProjection() const ;
    glm::mat4 getPerspective() const ;
    glm::mat4 getOrtho() const ;
    glm::mat4 getFrustum() const ;

    void getFrustumVert(std::vector<glm::vec4>& corners);

    std::string desc() const ;

} ;





