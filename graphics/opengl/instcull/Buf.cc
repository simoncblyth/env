
#include "Buf.hh"

Buf::Buf(unsigned num_items_, unsigned num_bytes_, void* ptr_)
    :
    id(-1),
    num_items(num_items_),
    num_bytes(num_bytes_),
    ptr(ptr_)
{
}
 
