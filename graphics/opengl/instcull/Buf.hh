#pragma once

struct Buf
{
    int id ; 
    unsigned num_bytes ;
    void* ptr ;
    Buf(unsigned num_bytes_, void* ptr_) ;
};



