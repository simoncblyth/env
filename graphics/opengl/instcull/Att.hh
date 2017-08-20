#pragma once


struct Prog ; 

struct Att
{
   const Prog& prog ; 
   GLuint loc ; 
   Att(const Prog& prog_ , unsigned loc_, unsigned ncomp_, unsigned stride_, unsigned offset_);
};


struct Q
{
    const char* name ; 
    GLenum      key ; 
    GLint       val ; 

    Q(const char* name_, GLenum key_ );

};


