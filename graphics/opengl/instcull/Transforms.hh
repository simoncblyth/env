#pragma once

#include "DEMO_API_EXPORT.hh"

struct DEMO_API Transforms
{
    unsigned ni ; 
    unsigned nj ; 
    unsigned nk ; 

    float* itra ; 

    Transforms(unsigned ni_, unsigned nj_, unsigned nk_, float* itra_=NULL ) ;
    void mockup();
    void mockup_spiral( glm::mat4& m , float fr );
    void mockup_diagonal( glm::mat4& m , float fr );

    void dump(unsigned n=0);

    unsigned num_items() const ;
    unsigned num_bytes() const ;
    unsigned num_floats() const ;

};

