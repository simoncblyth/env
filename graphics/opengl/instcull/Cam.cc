#include <iostream>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Cam.hh"


std::string Cam::desc()
{
    std::stringstream ss ; 

    ss 
        << " near " << near << std::endl 
        << " far " << far << std::endl 
        << " zoom " << zoom << std::endl 
        << " aspect " << aspect << std::endl 
        << " parallel " << parallel << std::endl 
        << " left " << getLeft() << std::endl 
        << " right " << getRight() << std::endl 
        << " bottom " << getBottom() << std::endl 
        << " top " << getTop() << std::endl 
        ;

    return ss.str();
}


Cam::Cam(int width, int height, float basis )
    :
    zoom(1.f),
    parallel(false),
    parascale(1.f)
{
    setSize(width, height);
    setFocus(basis);
}

void Cam::setSize(int width, int height )
{
    size[0] = width ;
    size[1] = height ;
    aspect  = (float)width/(float)height ;   // (> 1 for landscape) 
}

void Cam::setFocus(float basis_ )
{
    basis = basis_ ; 
    near = basis/10.f ; 
    far = basis*5.f ; 
}


float Cam::getScale(){ return parallel ? parascale  : near ; }
float Cam::getTop(){    return getScale() / zoom ; }
float Cam::getBottom(){ return -getScale() / zoom ; }
float Cam::getLeft(){   return -aspect * getScale() / zoom ; }
float Cam::getRight(){  return  aspect * getScale() / zoom ; }

void Cam::setYfov(float yfov_deg)
{
    // setYfov(90.) -> setZoom(1.)
    // fov = 2atan(1/zoom)
    // zoom = 1/tan(fov/2)

    zoom = 1.f/glm::tan(yfov_deg*0.5f*glm::pi<float>()/180.f );
}

float Cam::getYfov()
{
    float yfov_deg = 2.f*glm::atan(1.f/zoom)*180.f/glm::pi<float>() ;
    return yfov_deg ;
}

glm::mat4 Cam::getPerspective()  // seems not used
{
    return glm::perspective(getYfov(), aspect, near, far);
}

glm::mat4 Cam::getProjection()
{
    return parallel ? getOrtho() : getFrustum() ;
}
glm::mat4 Cam::getOrtho()
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), near, far );
}
glm::mat4 Cam::getFrustum()
{
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), near, far );
}




