#include <cassert>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Tra.hh"
#include "Buf.hh"
#include "BB.hh"
#include "UV.hh"
#include "Sphere.hh"


const unsigned Tra::QSIZE = sizeof(float)*4 ; 

unsigned Tra::num_items() const 
{
    return ni ;
}
unsigned Tra::num_floats() const 
{
    return ni*nj*nk ;
}
unsigned Tra::num_bytes() const 
{
    return sizeof(float)*num_floats() ; 
}


Tra::Tra(unsigned ni_, char shape_) 
    : 
    ni(ni_), 
    nj(4), 
    nk(4), 
    buf(NULL),
    bb(NULL)
{
    std::vector<glm::mat4> mat ; 
    mockup(mat, shape_);        
    populate(mat);
}


Tra::Tra(const std::vector<glm::mat4>& mat) 
    :
    ni(mat.size()),
    nj(4),
    nk(4),
    buf(NULL),
    bb(NULL) 
{
    populate(mat);
}

void Tra::populate(const std::vector<glm::mat4>& mat)
{
    buf = Buf::Make(mat);
    bb = BB::FromMat(mat);
    ce = bb->get_center_extent();
}


glm::vec3 Tra::SpherePos(const UV& uv, float radius)
{
    glm::vec3 pos(0.f);
    Sphere::_par_pos_body(pos,  uv,  radius ) ;
    return pos ; 
}

Tra* Tra::MakeGlobe(float radius, unsigned nu, unsigned nv ) 
{
    std::vector<glm::mat4> mat ; 
    unsigned s = 0 ; 

    glm::vec3 axis(0,0,1);
    glm::vec3 scal(0.9);

    for(unsigned v=0 ; v < nv ; v++)  // polar
    { 
        bool is_pole = v == 0 || v == nv - 1 ; 
        unsigned unu = is_pole ? 1 : nu ;    // only one position at poles

        for(unsigned u=0 ; u < unu; u++)  // azimuthal
        {
            UV uv = make_UV(s,u+0,v+0,nu,nv, 0);

            glm::mat4 m(1.f);
            float angle = uv.fu2pi() ;

            glm::vec3 tlat = SpherePos(uv, radius);

            m = glm::scale(m, scal) ; 
            m = glm::translate( m, tlat ) ;
            m = glm::rotate(m, angle , axis) ; 

            mat.push_back(m);          
        }
    }
    Tra* tra = new Tra(mat);
    return tra ; 
}



void Tra::mockup(std::vector<glm::mat4>& mat, char shape)
{ 
    for(unsigned i=0 ; i < ni ; i++)
    {
        float fr = float(i)/float(ni)  ;  // 0->1  
        glm::mat4 m(1.f) ;
        switch(shape)
        {
           case 'S': mockup_spiral(m, fr ); break ; 
           case 'D': mockup_diagonal(m, fr ); break ;
           case 'I': break ;
       } 
       mat.push_back(m);          
    }
}

void Tra::mockup_spiral( glm::mat4& m , float fr )
{
    glm::vec3 axis(0,0,1);
    glm::vec3 scal(0.9);

    float angle = glm::pi<float>()*2.f*fr ;  // 0->2pi
    glm::vec3 tlat( fr*glm::cos(3*angle),fr*glm::sin(3*angle),0);

    m = glm::scale(m, scal) ; 
    m = glm::translate(m, tlat );   // curious to get expected mrix form need to translate first
    m = glm::rotate(m, angle , axis) ; 
}

void Tra::mockup_diagonal( glm::mat4& m , float fr )
{
    float f = 2.f*fr - 1.0f ;  // -1:1  
    glm::vec3 tlat( f, f, 0);
    m = glm::translate(m, tlat );   // curious to get expected mrix form need to translate first
}



void Tra::dump(unsigned n)
{
    unsigned ndump = n > 0 ? n : ni ; 

    std::cout << "Tra::dump " 
              << " n " << n 
              << " ni " << ni
              << " ndump " << ndump
              << std::endl
              ;

    std::cout << "Tra::dump " 
              << " bb " << bb->desc()
              << std::endl
              ; 

 
    float* data = (float*)buf->ptr;
    for(unsigned i=0 ; i < ndump ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++)
        {
            for(unsigned k=0 ; k < nk ; k++)   
            {
                unsigned idx = i*nj*nk + j*nk + k ;
                std::cout << std::setw(10) << data[idx] << " " ;                
            }
            std::cout << std::endl  ; 
        }
        std::cout << std::endl  ; 
        std::cout << std::endl  ; 
    }
}



