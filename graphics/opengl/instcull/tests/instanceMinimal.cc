/*
Fairly Minimal Example of OpenGL Instancing 
===============================================

Using glVertexAttribDivisor, glDrawArraysInstanced
    
*/

#include <vector>
#include <iostream>
#include <cassert>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Renderer.hh"

const char* vertSrc = R"glsl(

    #version 400 core
    layout(location = 0) in vec4 vPosition ;
    layout(location = 1) in vec4 iPosition ;
    void main()
    {
        //gl_Position = vec4( vPosition.x, vPosition.y, vPosition.z, 1.0 ) ;  
        gl_Position = vec4( vPosition.x + iPosition.x, vPosition.y + iPosition.y, vPosition.z + iPosition.z, 1.0 ) ;  
    }
)glsl";


const char* fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";

struct V { float x,y,z,w ; };
static const unsigned NUM_VPOS = 3 ; 

V vpos[NUM_VPOS] = 
{
    { -0.1f , -0.1f,  0.f,  1.f }, 
    { -0.1f ,  0.1f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};

static const unsigned NUM_IPOS = 8 ; 
V ipos[NUM_IPOS] = 
{
    {   0.1f ,   0.1f,   0.f,  1.f }, 
    {   0.2f ,   0.2f,   0.f,  1.f },
    {   0.3f ,   0.3f,   0.f,  1.f },
    {   0.4f ,   0.4f,   0.f,  1.f },
    {  -0.1f ,  -0.1f,   0.f,  1.f }, 
    {  -0.2f ,  -0.2f,   0.f,  1.f },
    {  -0.3f ,  -0.3f,   0.f,  1.f },
    {  -0.4f ,  -0.4f,   0.f,  1.f }
};


int main()
{
    Frame frame ; 
    Prog prog(vertSrc, NULL, fragSrc ) ; 
    prog.compile();
    prog.create();
    prog.link();

    Buf v( NUM_VPOS, sizeof(vpos),vpos ) ; 
    Buf i( NUM_IPOS, sizeof(ipos),ipos ) ; 

    Renderer rdr ; 
    rdr.upload(&v, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    rdr.upload(&i, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLint vPosition = prog.getAttribLocation("vPosition");
    glBindBuffer(GL_ARRAY_BUFFER, v.id);
    glEnableVertexAttribArray(vPosition);
    glVertexAttribPointer(vPosition, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float),  NULL);
    
    GLint iPosition = prog.getAttribLocation("iPosition");
    glBindBuffer(GL_ARRAY_BUFFER, i.id);
    glEnableVertexAttribArray(iPosition);
    glVertexAttribPointer(iPosition, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float),  NULL);
    glVertexAttribDivisor(iPosition, 1 );

    while (!glfwWindowShouldClose(frame.window))
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArraysInstanced(GL_TRIANGLES, 0, NUM_VPOS, NUM_IPOS );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    prog.destroy();
    rdr.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


