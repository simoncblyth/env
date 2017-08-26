#pragma once

#include "DEMO_API_EXPORT.hh"
#include "Prim.hh"

struct DEMO_API Cube : Prim 
{
    float sx ; 
    float sy ; 
    float sz ; 
    float cx ; 
    float cy ; 
    float cz ; 

    Cube( float sx_=1.f, float sy_=1.f, float sz_=1.f , float cx_=0.f , float cy_=0.f , float cz=0.f );
    void init();

};



