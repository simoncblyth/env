
#include <sstream>
#include "Buf.hh"

Buf::Buf(unsigned num_items_, unsigned num_bytes_, void* ptr_)
    :
    id(-1),
    num_items(num_items_),
    num_bytes(num_bytes_),
    ptr(ptr_)
{
}
    
std::string Buf::desc()
{
    std::stringstream ss ; 

    ss << "Buf"
       << " id " << id  
       << " num_items " << num_items  
       << " num_bytes " << num_bytes
       << " num_bytes/num_items " << num_bytes/num_items
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

 
