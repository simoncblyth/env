#pragma once

struct Att
{
   GLuint loc ; 
   Att(unsigned loc_, unsigned ncomp_, unsigned stride_, unsigned offset_);
};


struct Q
{
    const char* name ; 
    GLenum      key ; 
    GLint       val ; 

    Q(const char* name_, GLenum key_ );

};


