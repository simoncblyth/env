#pragma once

struct Transforms
{
    unsigned ni ; 
    unsigned nj ; 
    unsigned nk ; 

    float* itra ; 

    Transforms(unsigned ni_, unsigned nj_, unsigned nk_, float* itra_=NULL ) ;
    void mockup();
    void dump();

    unsigned num_items() const ;
    unsigned num_bytes() const ;
    unsigned num_floats() const ;

};

