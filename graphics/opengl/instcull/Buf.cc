
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
 
