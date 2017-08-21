#pragma once

#include "DEMO_API_EXPORT.hh"

struct DEMO_API Buf
{
    unsigned id ; 
    unsigned num_bytes ;
    void* ptr ;
    Buf(unsigned num_bytes_, void* ptr_) ;
};



