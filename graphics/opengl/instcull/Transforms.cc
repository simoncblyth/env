#include <iostream>
#include <iomanip>

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

    for(unsigned i=0 ; i < ni ; i++)
    for(unsigned j=0 ; j < nj ; j++)
    for(unsigned k=0 ; k < nk ; k++)   
    {
        unsigned idx = i*nj*nk + j*nk + k ;
        itra[idx] = k == j ? 1.f : 0.f ; 

        if(j == 3 && k == 0) itra[idx] = 2.f*float(i)/float(ni) - 1.0f  ; // x-offset         
        if(j == 3 && k == 1) itra[idx] = 2.f*float(i)/float(ni) - 1.0f  ; // y-offset         
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



