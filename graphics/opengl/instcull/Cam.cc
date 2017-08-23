#include <iostream>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "G.hh"
#include "Cam.hh"


std::string Cam::desc() const 
{
    std::stringstream ss ; 
    glm::mat4 projection = getProjection() ;


    ss 
        << " basis " << basis << std::endl 
        << " factor " << factor << std::endl 
        << " near " << near << std::endl 
        << " far " << far << std::endl 
        << " zoom " << zoom << std::endl 
        << " aspect " << getAspect() << std::endl 
        << " parallel " << parallel << std::endl 
        << " left " << getLeft() << std::endl 
        << " right " << getRight() << std::endl 
        << " bottom " << getBottom() << std::endl 
        << " top " << getTop() << std::endl 
        << " yfov " << getYfov() 
        << std::endl 
        << G::gpresent("proj", projection )
        << std::endl 
        ;


    return ss.str();
}


Cam::Cam(int width_, int height_, float basis )
    :
    zoom(1.f),
    parallel(false),
    parascale(1.f)
{
    setSize(width_, height_);
    setFocus(basis);
}

void Cam::setSize(int width_, int height_ )
{
    width = width_ ;
    height = height_ ;
}

void Cam::setFocus(float basis_, float factor_)
{
    basis = basis_ ; 
    factor = factor_ ; 
    near = basis/factor_ ; 
    far = basis*factor_ ; 
}


float Cam::getAspect() const { return (float)width/(float)height ; }  //  (> 1 for landscape) 
float Cam::getScale() const { return parallel ? parascale  : near ; }
float Cam::getTop() const {    return getScale() / zoom ; }
float Cam::getBottom() const { return -getScale() / zoom ; }
float Cam::getLeft() const {   return -getAspect() * getScale() / zoom ; }
float Cam::getRight() const {  return  getAspect() * getScale() / zoom ; }

void Cam::setYfov(float yfov_deg)
{
    // setYfov(90.) -> setZoom(1.)
    // fov = 2atan(1/zoom)
    // zoom = 1/tan(fov/2)

    zoom = 1.f/glm::tan(yfov_deg*0.5f*glm::pi<float>()/180.f );
}

float Cam::getYfov() const 
{
    float yfov_deg = 2.f*glm::atan(1.f/zoom)*180.f/glm::pi<float>() ;
    return yfov_deg ;
}

glm::mat4 Cam::getPerspective() const   // seems not used
{
    return glm::perspective(getYfov(), getAspect(), near, far);
}

glm::mat4 Cam::getProjection() const 
{
    return parallel ? getOrtho() : getFrustum() ;
}
glm::mat4 Cam::getOrtho() const 
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), near, far );
}
glm::mat4 Cam::getFrustum() const 
{ 
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), near, far );
}

