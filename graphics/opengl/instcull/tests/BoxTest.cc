#include <iostream>
#include <cassert>

#include "Box.hh"
#include "Buf.hh"
#include "G.hh"

int main()
{
    Box box(0,10.f);
 
    unsigned num_vert = box.tri_vert.size() ;

    assert( num_vert > 0 );

    for(unsigned i=0 ; i < num_vert ; i++)
    {
        const glm::vec4& p = box.tri_vert[i]; 
        std::cout << G::gpresent("p", p ) << std::endl ; 
    }

    Buf* buf = box.buf();

    std::cout << buf->desc() << std::endl ; 



    return 0 ; 
}
