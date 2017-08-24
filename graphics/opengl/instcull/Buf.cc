
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



Buf* Buf::Make(const std::vector<glm::vec4>& vert) 
{ 
    
    unsigned num_vert = vert.size();
    unsigned num_float = num_vert*4 ; 
    unsigned num_byte = num_float*sizeof(float) ; 

    float* dest = new float[num_float] ; 
    memcpy(dest, vert.data(), num_byte ) ; 

    return new Buf( num_vert, num_byte, (void*)dest ) ; 
} 



 
