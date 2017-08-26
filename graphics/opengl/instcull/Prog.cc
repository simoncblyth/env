#include <cassert>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"



// http://antongerdelan.net/opengl/shaders.html

const char* Prog::ShaderType(GLenum type)
{
    const char* s = NULL;
    switch(type)
    {
        case GL_VERTEX_SHADER  : s = "vertex"   ; break;
        case GL_GEOMETRY_SHADER: s = "geometry" ; break;
        case GL_FRAGMENT_SHADER: s = "fragment" ; break;
    }
    return s ; 
}

Prog::Prog(const char* vertSrc_, const char* geomSrc_,  const char* fragSrc_)
        :
        vertSrc(vertSrc_),
        geomSrc(geomSrc_),
        fragSrc(fragSrc_),
        vert(vertSrc != NULL),
        geom(geomSrc != NULL),
        frag(fragSrc != NULL)
{
}

void Prog::compile()
{
    if(vert) vertShader = compile(GL_VERTEX_SHADER,  vertSrc ) ;
    if(geom) geomShader = compile(GL_GEOMETRY_SHADER, geomSrc ) ;
    if(frag) fragShader = compile(GL_FRAGMENT_SHADER, fragSrc );
}

unsigned Prog::compile(GLenum type, const char* src)
{
    std::cerr << "\n\n//////////// Compile " <<  ShaderType(type) << "\n" ;
    std::cerr << src << "\n" ; 

    unsigned shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        
        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        std::cerr << "Compile failure in shader: " <<  ShaderType(type) << "\n" <<  strInfoLog << "\n" ;
        delete[] strInfoLog;
    }
    else
    {
        std::cerr << "Compile OK shader: " <<  ShaderType(type) << "\n" ;
    }
    assert(status == GL_TRUE) ;
    return shader ; 
}


void Prog::create()
{
    program = glCreateProgram();
    if(vert) glAttachShader(program, vertShader);
    if(geom) glAttachShader(program, geomShader);
    if(frag) glAttachShader(program, fragShader);
}

void Prog::link()
{
    glLinkProgram(program);

    GLint status;
    glGetProgramiv (program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        
        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
        std::cerr <<  "Linker failure: \n" <<  strInfoLog << "\n"  ;
        delete[] strInfoLog;
    }

    glUseProgram(program);
} 


void Prog::destroy()
{
    glDeleteProgram(program);
    if(geom) glDeleteShader(geomShader);
    if(vert) glDeleteShader(vertShader);
    if(frag) glDeleteShader(fragShader);
}

int Prog::getAttribLocation(const char* att) const 
{
    return glGetAttribLocation(program, att );
}




