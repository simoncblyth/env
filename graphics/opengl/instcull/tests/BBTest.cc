#include <cassert>
#include "BB.hh"

int main()
{
    BB bb0 ; 
    assert( bb0.is_empty() ); 

    BB bb1(1.f) ; 
    assert( !bb1.is_empty() ); 
 
    return 0 ;  
}

