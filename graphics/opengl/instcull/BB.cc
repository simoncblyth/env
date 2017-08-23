

#include <sstream>
#include "G.hh"
#include "BB.hh"

#include <glm/gtx/component_wise.hpp>


BB::BB(float extent)
   :
   min(-extent),
   max(extent)
{
}



void BB::include(const glm::vec3& p)
{
    if(is_empty())
    {   
        min = p ; 
        max = p ; 
    }   
    else
    {   
        min = glm::min( min, p );
        max = glm::max( max, p );
    }   
}


bool BB::is_empty() const 
{   
    return min.x == 0. && min.y == 0. && min.z == 0. && max.x == 0. && max.y == 0. && max.z == 0.  ;   
}   

void BB::set_empty()
{   
    min.x = 0. ; 
    min.y = 0. ; 
    min.z = 0. ; 

    max.x = 0. ; 
    max.y = 0. ; 
    max.z = 0. ; 
}   

std::string BB::desc() const 
{
    std::stringstream ss ; 


    glm::vec4 ce = get_center_extent() ;

    ss 
        << G::gpresent("min", min) 
        << " " 
        << G::gpresent("max", max) 
        << " " 
        << G::gpresent("ce", ce) 
        << " " 
        ;

    return ss.str();
}


glm::vec4 BB::get_center_extent() const 
{
    glm::vec4 ce ; 
    glm::vec3 dim =  max - min ; 
    glm::vec3 cen =  max + min ; 
      
    dim /= 2. ; 
    cen /= 2. ; 

    float extent = glm::compMax(dim);

    ce.x = cen.x ; 
    ce.y = cen.y ; 
    ce.z = cen.z ; 
    ce.w = extent ; 

    return ce ; 
}



