#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <glm/gtc/matrix_transform.hpp>

#include "G.hh"
#include "Comp.hh"
#include "Cam.hh"
#include "Vue.hh"


Comp::Comp()
    :
    vue(new Vue),
    cam(new Cam)
{
}


void Comp::setCenterExtent(float x, float y, float z, float w)
{
    glm::vec4 ce(x,y,z,w);
    setCenterExtent(ce);
}

void Comp::setCenterExtent(const glm::vec4& ce)
{
    center_extent = ce ; 

    glm::vec3 tr(ce.x, ce.y, ce.z);
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);

    model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);     // translate then blowup full size
    world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);   // shrink then translate 

}


void Comp::update()
{

    projection = cam->getProjection();

    vue->getTransforms(model2world, world2camera, camera2world, gaze );  // model2world -> others 

    world2eye = world2camera ; 
    world2clip = projection * world2eye ;    //  ModelViewProjection
}



glm::vec3 Comp::getNDC(const glm::vec4& world)
{
    const glm::vec4 eye = world2eye * world ;
    glm::vec4 clip = projection * eye ;
    glm::vec3 ndc( clip );
    ndc /= clip.w ;  
    return ndc ; 
}

glm::vec3 Comp::getNDC2(const glm::vec4& world)
{
    glm::vec4 clip = world2clip * world ;
    glm::vec3 ndc( clip );
    ndc /= clip.w ;  
    return ndc ; 

}




std::string Comp::desc()
{
    std::stringstream ss ;   
    ss << G::gpresent( "center_extent", center_extent ) << std::endl << std::endl ;
    ss << G::gpresent( "model2world", model2world ) << std::endl ;
    ss << G::gpresent( "world2model", world2model ) << std::endl ;
    ss << G::gpresent( "projection",  projection ) << std::endl ;
    ss << G::gpresent( "world2camera", world2camera ) << std::endl ;
    ss << G::gpresent( "camera2world", camera2world ) << std::endl ;
    ss << G::gpresent( "gaze", center_extent ) << std::endl << std::endl ;
    ss << G::gpresent( "world2clip", world2clip ) << std::endl ;
    ss << G::gpresent( "world2eye", world2eye ) << std::endl ;

    ss << "Vue" << std::endl << vue->desc() ; 
    ss << "Cam" << std::endl << cam->desc() ; 
    
    return ss.str() ;
}

void Comp::dump()
{
    std::cout << desc() << std::endl ;

    dumpCorners() ;
}


void Comp::dumpCorners()
{
    std::vector<glm::vec4> world ;

    glm::vec4& ce = center_extent ; 



    world.push_back( { ce.x       , ce.y        , ce.z       , 1.0 } );

    for(unsigned i=0 ; i < 8 ; i++)
    {
        world.push_back( 
            { 
              i & 1 ? ce.x + ce.w : ce.x - ce.w ,
              i & 2 ? ce.y + ce.w : ce.y - ce.w ,
              i & 4 ? ce.z + ce.w : ce.z - ce.w ,
              1.f
            });
    }

   for(unsigned i=0 ; i < world.size() ; i++)
   {
       const glm::vec4& wpos = world[i] ;

       glm::vec3 ndc = getNDC(wpos) ; 
       glm::vec3 ndc2 = getNDC2(wpos) ; 
  
       std::cout
               << G::gpresent("world",  wpos )
               << std::endl 
               << G::gpresent("model",  world2model  * wpos )
               << std::endl 
               << G::gpresent("camera", world2camera * wpos )
               << std::endl 
               << G::gpresent("eye",    world2eye * wpos )
               << std::endl 
               << G::gpresent("clip",   world2clip * wpos )
               << std::endl 
               << G::gpresent("ndc",    ndc )
               << std::endl 
               << G::gpresent("ndc2",   ndc2 )
               << std::endl 
               << std::endl 
               ;

   }
}




