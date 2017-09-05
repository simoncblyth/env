#include <sstream>
#include <cstddef>
#include "Buf.hh"
#include "Buf4.hh"

Buf4::Buf4()
    :
    x(NULL),
    y(NULL),
    z(NULL),
    w(NULL),
    devnull(NULL)
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


std::string Buf4::desc() const 
{
    std::stringstream ss ; 

    ss << "Buf4"  
       << " num_buf " << num_buf()
       << " x " << ( x ? x->brief() : "-" )
       << " y " << ( y ? y->brief() : "-" )
       << " z " << ( z ? z->brief() : "-" )
       << " w " << ( w ? w->brief() : "-" )
       << " devnull " << ( devnull ? devnull->brief() : "-" )
       ;

    return ss.str();
}


void Buf4::dump() const 
{
    unsigned nb = num_buf();
    for(int i=0 ; i < nb ; i++) 
    {
         std::cout << " i " << i 
                   << " at(i)->id  " << at(i)->id 
                   << std::endl ; 
 
    }
}



