#pragma once

#include "DEMO_API_EXPORT.hh"

struct DEMO_API V { float x,y,z,w ; };

struct Buf ; 

struct DEMO_API Pos
{
    enum { NUM_VPOS = 3, NUM_INST = 8 } ; 

    static const V apos[NUM_VPOS] ; 
    static const V bpos[NUM_VPOS] ; 
    static const V ipos[NUM_INST] ; 
    static const V jpos[NUM_INST] ; 
   // static const V onetri[3] ; 

    static Buf* a();
    static Buf* b();
    static Buf* i();
    static Buf* j();
    static Buf* onetriangle(float x=0.5f, float y=0.5f, float z=-1.f);

};


