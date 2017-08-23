#pragma once

#include <string>
#include "DEMO_API_EXPORT.hh"

struct DEMO_API Buf
{
    unsigned id ; 
    unsigned num_items ;
    unsigned num_bytes ;
    void*    ptr ;
    Buf(unsigned num_items_, unsigned num_bytes_, void* ptr_) ;

    std::string desc();

};



