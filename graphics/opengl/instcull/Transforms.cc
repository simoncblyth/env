#include <iostream>
#include <iomanip>


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Transforms.hh"


Transforms::Transforms(unsigned ni_, unsigned nj_, unsigned nk_, float* itra_ ) : ni(ni_), nj(nj_), nk(nk_), itra(itra_)
{
    if(itra == NULL) mockup();        
}


unsigned Transforms::num_items() const 
{
    return ni ;
}
unsigned Transforms::num_floats() const 
{
    return ni*nj*nk ;
}
unsigned Transforms::num_bytes() const 
{
    return sizeof(float)*num_floats() ; 
}


void Transforms::mockup()
{
    itra = new float[ni*nj*nk] ;

    glm::vec3 axis(0,0,1);
    glm::vec3 scal(0.9);
    bool transpose = false ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        float fr = float(i)/float(ni)  ;  // 0->1  
        //float f = 2.f*fr - 1.0f ;  // -1:1  

        float angle = glm::pi<float>()*2.f*fr ;  // 0->2pi
        glm::vec3 tlat( fr*glm::cos(3*angle),fr*glm::sin(3*angle),0);

        glm::mat4 mat(1.f) ;
        mat = glm::scale(mat, scal) ; 
        mat = glm::translate(mat, tlat );   // curious to get expected matrix form need to translate first
        mat = glm::rotate(mat, angle , axis) ; 

        for(unsigned j=0 ; j < nj ; j++)
        {
            for(unsigned k=0 ; k < nk ; k++)   
            {
                unsigned idx = i*nj*nk + j*nk + k ;
                itra[idx] = transpose ? mat[k][j] : mat[j][k] ; 
            }
        }
    }   
}


void Transforms::dump()
{
    for(unsigned i=0 ; i < ni ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++)
        {
            for(unsigned k=0 ; k < nk ; k++)   
            {
                unsigned idx = i*nj*nk + j*nk + k ;
                std::cout << std::setw(10) << itra[idx] << " " ;                
            }
            std::cout << std::endl  ; 
        }
        std::cout << std::endl  ; 
        std::cout << std::endl  ; 
    }
}



