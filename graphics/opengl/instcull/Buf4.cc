#include <cstddef>
#include "Buf4.hh"

Buf4::Buf4()
    :
    x(NULL),
    y(NULL),
    z(NULL),
    w(NULL)
{
}

Buf* Buf4::at(unsigned i) const 
{
    Buf* b = NULL ; 
    switch(i)
    {
        case 0: b = x ; break ; 
        case 1: b = y ; break ; 
        case 2: b = z ; break ; 
        case 3: b = w ; break ; 
    }  
    return b ; 
}

unsigned Buf4::num_buf() const 
{
    return ( x ? 1 : 0 ) + ( y ? 1 : 0 ) + ( z ? 1 : 0 ) + ( w ? 1 : 0 ) ;
}
