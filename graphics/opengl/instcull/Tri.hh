#pragma once

#include "DEMO_API_EXPORT.hh"
#include "Prim.hh"

struct DEMO_API Tri : Prim 
{
    float sx ; 
    float sy ; 
    float sz ; 
    float cx ; 
    float cy ; 
    float cz ; 

    Tri( float sx_, float sy_, float sz_ , float cx_ , float cy_ , float cz );
    void init();

};



