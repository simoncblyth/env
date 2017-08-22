#pragma once

#include "DEMO_API_EXPORT.hh"
#include <string>

struct DEMO_API Cam
{
    int size[2];  
    float aspect ; 
    float basis ; 
    float near ; 
    float far ; 
    float zoom ; 

    bool parallel ; 
    float parascale ; 

    Cam( int width=1024, int height=768, float basis=1000.f );

    void setSize(int width, int height );
    void setFocus(float basis);
    void setYfov(float yfov_deg);

    float getScale(); 
    float getTop();
    float getBottom();
    float getLeft();
    float getRight();
    float getYfov();

    glm::mat4 getProjection();
    glm::mat4 getPerspective();
    glm::mat4 getOrtho();
    glm::mat4 getFrustum();

    std::string desc();

};





