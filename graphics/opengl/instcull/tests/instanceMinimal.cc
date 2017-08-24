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
#include "Pos.hh"
#include "Box.hh"

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


void upload(Buf* buf, GLenum target, GLenum usage )
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(target, buf->id);
    glBufferData(target, buf->num_bytes, buf->ptr, usage);
    glBindBuffer(target, 0);
}

int main()
{
    Frame frame ; 
    Prog prog(vertSrc, NULL, fragSrc ) ; 
    prog.compile();
    prog.create();
    prog.link();

    Box box(0, 0.05f );

    //Buf* a = Pos::a();
    Buf* a = box.buf();

    Buf* i = Pos::i();

    GLuint vao ; 
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    upload(a, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    upload(i, GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    GLint vPosition = prog.getAttribLocation("vPosition");
    glBindBuffer(GL_ARRAY_BUFFER, a->id);
    glEnableVertexAttribArray(vPosition);
    glVertexAttribPointer(vPosition, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float),  NULL);
    
    GLint iPosition = prog.getAttribLocation("iPosition");
    glBindBuffer(GL_ARRAY_BUFFER, i->id);
    glEnableVertexAttribArray(iPosition);
    glVertexAttribPointer(iPosition, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float),  NULL);
    glVertexAttribDivisor(iPosition, 1 );


    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(frame.window))
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        glDrawArraysInstanced(GL_TRIANGLES, 0, a->num_items, i->num_items );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    prog.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


