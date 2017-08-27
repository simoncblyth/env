#pragma once

#include <string>
#include "DEMO_API_EXPORT.hh"

struct Buf ; 

struct DEMO_API Buf4
{
    Buf* x ;
    Buf* y ;
    Buf* z ;
    Buf* w ;

    Buf* devnull ;


    Buf4();

    Buf* at(unsigned i) const ;
    unsigned num_buf() const ;

    std::string desc() const ;


};
 



