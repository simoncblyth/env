#pragma once

#include "DEMO_API_EXPORT.hh"

struct Buf ; 

struct DEMO_API Buf4
{
    Buf* x ;
    Buf* y ;
    Buf* z ;
    Buf* w ;

    Buf4();

    Buf* at(unsigned i) const ;
    unsigned num_buf() const ;
};
 



