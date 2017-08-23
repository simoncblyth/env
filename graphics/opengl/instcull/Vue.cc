#include "G.hh"
#include "Vue.hh"

#include <iostream>
#include <sstream>
#include <iomanip>


#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>


Vue::Vue()
{
    home();
}   

void Vue::home()
{
    setEye(-1.f,-1.f, 0.f);
    setLook(0.f, 0.f, 0.f);
    setUp(  0.f, 0.f, 1.f);
}

void Vue::setEye(float x, float y, float z)
{
    eye.x = x ; 
    eye.y = y ; 
    eye.z = z ;
    eye.w = 1.f ;  
}
void Vue::setLook(float x, float y, float z)
{
    look.x = x ; 
    look.y = y ; 
    look.z = z ;
    look.w = 1.f ;  
}
void Vue::setUp(float x, float y, float z)
{
    up.x = x ; 
    up.y = y ; 
    up.z = z ;
    up.w = 0.f ;  // direction, not position    
}



void Vue::getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze )
{
    /*  
    See 
           env/geant4/geometry/collada/g4daeview/daeutil.py
           env/graphics/glm/lookat.cc


    OpenGL eye space convention with forward as -Z
    means that have to negate the forward basis vector in order 
    to create a right-handed coordinate system.

    Construct matrix using the normalized basis vectors::    

                             -Z
                       +Y    .  
                        |   .
                  EY    |  .  -EZ forward 
                  top   | .  
                        |. 
                        E-------- +X
                       /  EX right
                      /
                     /
                   +Z

    */

    glm::vec3 eye_w   = glm::vec3(m2w * eye );
    glm::vec3 up_w    = glm::vec3(m2w * up );
    glm::vec3 gze_w  = glm::vec3(m2w * (look - eye));  

    glm::vec3 forward = glm::normalize(gze_w);                        // -Z
    glm::vec3 right   = glm::normalize(glm::cross(forward,up_w));     // +X
    glm::vec3 top     = glm::normalize(glm::cross(right,forward));    // +Y

    glm::mat4 rot ;                      // orthogonal basis 
    rot[0] = glm::vec4( right, 0.f );    // +X
    rot[1] = glm::vec4( top  , 0.f );    // +Y
    rot[2] = glm::vec4( -forward, 0.f ); // +Z

    glm::mat4 origin_to_eye(glm::translate(glm::vec3(eye_w)));
    glm::mat4 eye_to_origin(glm::translate(glm::vec3(-eye_w)));  

    world2camera = glm::transpose(rot) * eye_to_origin  ;
    
    //  translate first putting the eye at the origin
    //  then rotate to point -Z forward
    //  this is equivalent to lookAt as used by OpenGL ModelVue

    camera2world = origin_to_eye * rot ;
    gaze = glm::vec4( gze_w, 0.f );

}



std::string Vue::desc()
{
    glm::mat4 m2w(1.f);
    glm::mat4 w2c ; 
    glm::mat4 c2w ; 
    glm::vec4 gze ; 
    getTransforms(m2w, w2c, c2w, gze) ; 

    std::stringstream ss; 
    ss << G::gpresent( "eye", eye ) << std::endl ;
    ss << G::gpresent( "look", look ) << std::endl ;
    ss << G::gpresent( "up", up ) << std::endl ;
    ss << G::gpresent( "m2w", m2w ) << std::endl ;
    ss << G::gpresent( "w2c", w2c ) << std::endl ;
    ss << G::gpresent( "gze", gze ) << std::endl ;

    return ss.str();
}

