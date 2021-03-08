// name=IMGTest ; stb- ; gcc $name.cc -lstdc++ -std=c++11 -I$(stb-dir) -I. -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>

#define IMG_IMPLEMENTATION 1 
#include "IMG.h"

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : nullptr ; 
    assert(path); 
    IMG img(path); 
    std::cout << img.desc() << std::endl ; 

    img.writePNG(); 
    img.writeJPG(100); 
    img.writeJPG(50); 
    img.writeJPG(10); 
    img.writeJPG(5); 
  
    return 0 ; 
}


