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


void Transforms::mockup_spiral( glm::mat4& mat , float fr )
{
    glm::vec3 axis(0,0,1);
    glm::vec3 scal(0.9);

    float angle = glm::pi<float>()*2.f*fr ;  // 0->2pi
    glm::vec3 tlat( fr*glm::cos(3*angle),fr*glm::sin(3*angle),0);

    mat = glm::scale(mat, scal) ; 
    mat = glm::translate(mat, tlat );   // curious to get expected matrix form need to translate first
    mat = glm::rotate(mat, angle , axis) ; 
}

void Transforms::mockup_diagonal( glm::mat4& mat , float fr )
{
    float f = 2.f*fr - 1.0f ;  // -1:1  
    glm::vec3 tlat( f, f, 0);
    mat = glm::translate(mat, tlat );   // curious to get expected matrix form need to translate first
}


void Transforms::mockup()
{
    itra = new float[ni*nj*nk] ;

    bool transpose = false ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        float fr = float(i)/float(ni)  ;  // 0->1  

        glm::mat4 mat(1.f) ;
        mockup_spiral(mat, fr );
        //mockup_diagonal(mat, fr );

          
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


void Transforms::dump(unsigned n)
{
    unsigned ndump = n > 0 ? n : ni ; 

    std::cout << "Transforms::dump " 
              << " n " << n 
              << " ni " << ni
              << " ndump " << ndump
              << std::endl
              ;
  

    for(unsigned i=0 ; i < ndump ; i++)
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



