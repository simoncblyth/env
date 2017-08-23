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

Buf* Pos::onetriangle(float x, float y, float z)  // (-x,-y,z,1) (-x,y,z,1) (x,0,z,1)
{ 
    std::vector<glm::vec4> tri ; 
    tri.push_back( { -x,   -y,  z , 1.f } );
    tri.push_back( { -x,    y,  z , 1.f } );
    tri.push_back( {  x,  0.0f, z , 1.f } );

    unsigned num_tri = tri.size();
    unsigned num_float = num_tri*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, tri.data(), num_byte ) ; 

    return new Buf( num_tri, num_byte, (void*)dest ) ; 
} 



