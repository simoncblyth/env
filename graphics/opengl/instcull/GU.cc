
#include <iostream>
#include <cassert>


#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include "GU.hh"


void GU::errchk(const char* msg)
{
    GLenum glError;
    if ((glError = glGetError()) != GL_NO_ERROR) 
     {
        std::cout 
            << msg 
            << " : Warning: OpenGL error code: " 
            << glError 
            << std::endl
            ;
    }   
}

