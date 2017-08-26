
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "G.hh"
#include "Buf.hh"

Buf::Buf(unsigned num_items_, unsigned num_bytes_, void* ptr_)
    :
    id(-1),
    num_items(num_items_),
    num_bytes(num_bytes_),
    ptr(ptr_)
{
}

unsigned Buf::item_bytes() const 
{
    assert( num_bytes % num_items == 0 );
    return num_bytes/num_items ; 
}


Buf* Buf::cloneEmpty() const 
{
    Buf* b = new Buf(num_items, num_bytes, NULL) ;
    return b ; 
}

void Buf::dump(const char* msg) const 
{
    std::cout << msg << std::endl ; 
    std::cout << desc() << std::endl ; 
    
    unsigned ib = item_bytes();

    if( ib == sizeof(unsigned) )
    {
        assert( num_items % 3 == 0 );
        unsigned num_tri = num_items/3 ; 
        for(unsigned i=0 ; i < num_tri ; i++ )
        {
             std::cout 
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 0) << " " 
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 1) << " "  
                << std::setw(5) <<  *((unsigned*)ptr + 3*i + 2) << " " 
                << std::endl 
                ; 
        }
        std::cout << std::endl ; 
    }
    else if( ib == sizeof(float)*4 )
    {
        for(unsigned i=0 ; i < num_items ; i++ ) 
        {        
             const glm::vec4& v = *((glm::vec4*)ptr + i ) ;  
             std::cout << G::gpresent("v",v) << std::endl ; 
        }
    }
    else if( ib == sizeof(float)*4*4 )
    {
        for(unsigned i=0 ; i < num_items ; i++ ) 
        {        
             const glm::mat4& m = *((glm::mat4*)ptr + i ) ;  
             std::cout << G::gpresent("m",m) << std::endl ; 
        }
    }
}

    
std::string Buf::desc() const 
{
    std::stringstream ss ; 

    ss << "Buf"
       << " id " << id  
       << " num_items " << num_items  
       << " num_bytes " << num_bytes
       << " item_bytes() " << item_bytes()
       ; 

    return ss.str();
}

void Buf::upload(GLenum target, GLenum usage )
{
    glGenBuffers(1, &this->id);
    glBindBuffer(target, this->id);
    glBufferData(target, this->num_bytes, this->ptr, usage);
    glBindBuffer(target, 0);
}

void Buf::uploadNull(GLenum target, GLenum usage )
{
    glGenBuffers(1, &this->id);
    glBindBuffer(target, this->id);
    glBufferData(target, this->num_bytes, NULL, usage);
    glBindBuffer(target, 0);
}


Buf* Buf::Make(const std::vector<glm::vec4>& vert) 
{     
    unsigned num_item = vert.size();
    unsigned num_float = num_item*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, vert.data(), num_byte ) ; 

    return new Buf( num_item, num_byte, (void*)dest ) ; 
} 

Buf* Buf::Make(const std::vector<glm::mat4>& mat) 
{     
    unsigned num_item = mat.size();
    unsigned num_float = num_item*4*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, mat.data(), num_byte ) ; 

    return new Buf( num_item, num_byte, (void*)dest ) ; 
} 

Buf* Buf::Make(const std::vector<unsigned>& elem) 
{     
    unsigned num_item = elem.size();
    unsigned num_unsigned = num_item ; 
    unsigned num_byte = num_unsigned*sizeof(unsigned) ; 

    unsigned* dest = new unsigned[num_unsigned] ; 
    memcpy(dest, elem.data(), num_byte ) ; 

    return new Buf( num_item, num_byte, (void*)dest ) ; 
} 

 
