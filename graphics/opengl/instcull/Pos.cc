#include <vector>
#include <glm/glm.hpp>


#include "Pos.hh"
#include "Buf.hh"


const V Pos::apos[NUM_VPOS] = 
{
    { -0.1f , -0.1f,  0.f,  1.f }, 
    { -0.1f ,  0.1f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};

const V Pos::bpos[NUM_VPOS] = 
{
    {  0.2f , -0.2f,  0.f,  1.f }, 
    {  0.2f ,  0.2f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};




const V Pos::ipos[NUM_INST] = 
{
    {   0.1f ,   0.1f,   0.f,  1.f }, 
    {   0.2f ,   0.2f,   0.f,  1.f },
    {   0.3f ,   0.3f,   0.f,  1.f },
    {   0.4f ,   0.4f,   0.f,  1.f },
    {  -0.1f ,  -0.1f,   0.f,  1.f }, 
    {  -0.2f ,  -0.2f,   0.f,  1.f },
    {  -0.3f ,  -0.3f,   0.f,  1.f },
    {  -0.4f ,  -0.4f,   0.f,  1.f }
};


const V Pos::jpos[NUM_INST] = 
{
    {   0.1f ,   -0.1f,   0.f,  1.f }, 
    {   0.2f ,   -0.2f,   0.f,  1.f },
    {   0.3f ,   -0.3f,   0.f,  1.f },
    {   0.4f ,   -0.4f,   0.f,  1.f },
    {  -0.1f ,    0.1f,   0.f,  1.f }, 
    {  -0.2f ,    0.2f,   0.f,  1.f },
    {  -0.3f ,    0.3f,   0.f,  1.f },
    {  -0.4f ,    0.4f,   0.f,  1.f }
};



Buf* Pos::a(){ return new Buf( NUM_VPOS, sizeof(apos),(void*)apos ) ; }  
Buf* Pos::b(){ return new Buf( NUM_VPOS, sizeof(bpos),(void*)bpos ) ; }  
Buf* Pos::i(){ return new Buf( NUM_INST, sizeof(ipos),(void*)ipos ) ; }  
Buf* Pos::j(){ return new Buf( NUM_INST, sizeof(jpos),(void*)jpos ) ; } 




