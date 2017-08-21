#include <iostream>
#include <iomanip>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Att.hh"


Att::Att(unsigned loc_, unsigned ncomp, unsigned stride_ , unsigned offset_)
    :
    loc(loc_)
{
   
    GLuint index = loc ;
 
    GLint  size = ncomp ;         // Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, 4.
    GLenum type = GL_FLOAT ;
    GLboolean normalized = GL_FALSE ;
    GLsizei stride = stride_ ;   // byte offset between consecutive generic vertex attributes, eg 4*sizeof(float) for vec4


    uintptr_t offset__ = offset_ ; 
    const GLvoid* offset = (void*)offset__  ;

    glVertexAttribPointer(index, size, type, normalized, stride, offset);
}

Q::Q(const char* name_, GLenum key_ ) : name(name_), key(key_)
{
    glGetIntegerv(key_, &val);
    std::cout 
           << " name " << std::setw(30) << name 
           << " val " << val  
           << " val/1e6 " << val/1e6  
           << std::endl;
}



