
#include <sstream>
#include "G.hh"
#include "Buf.hh"
#include "BB.hh"

// must be included after glm
#include <glm/gtx/component_wise.hpp>


BB::BB(float extent)
   :
   min(-extent),
   max(extent)
{
}



BB* BB::FromVert(const std::vector<glm::vec4>& vert)
{
    BB* bb = new BB ; 
    for(unsigned i=0 ; i < vert.size() ; i++) 
    {
        glm::vec3 p(vert[i]);
        bb->include(p);
    }
    return bb ; 
}  


BB* BB::FromMat(const std::vector<glm::mat4>& mat)
{
    BB* bb = new BB ; 
    for(unsigned i=0 ; i < mat.size() ; i++) 
    {
        const glm::mat4& m = mat[i] ;
        glm::vec3 p(m[3]);
        bb->include(p);
    }
    return bb ; 
} 

BB* BB::FromBuf(const Buf* buf )
{
    BB* bb = new BB ; 
    unsigned ib = buf->item_bytes();
    if( ib == sizeof(float)*4 )
    {
        for(unsigned i=0 ; i < buf->num_items ; i++ ) 
        {        
            const glm::vec4& v = *((glm::vec4*)buf->ptr + i ) ;  
            glm::vec3 p(v);
            bb->include(p);
        }
    }
    else if( ib == sizeof(float)*4*4 )
    {
        for(unsigned i=0 ; i < buf->num_items ; i++ ) 
        {        
            const glm::mat4& m = *((glm::mat4*)buf->ptr + i ) ;  
            glm::vec3 p(m[3]);
            bb->include(p);
        }
    }
    return bb ; 
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

    ss 
        << G::gpresent("min", min) 
        << " " 
        << G::gpresent("max", max) 
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


