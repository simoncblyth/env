#pragma once

#include "DEMO_API_EXPORT.hh"


struct DEMO_API Att
{
   GLuint loc ; 
   Att(unsigned loc_, unsigned ncomp_, unsigned stride_, unsigned offset_);
};


struct DEMO_API Q
{
    const char* name ; 
    GLenum      key ; 
    GLint       val ; 

    Q(const char* name_, GLenum key_ );

};


