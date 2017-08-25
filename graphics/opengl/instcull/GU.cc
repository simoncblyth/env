
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


void GU::ReplaceAll(std::string& subject, const char* search, const char* replace) 
{
    //https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) 
    {   
        subject.replace(pos, strlen(search), replace);
        pos += strlen(replace) ;
    }   
}




