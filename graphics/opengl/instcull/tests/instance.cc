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

void errcheck(const char* msg)
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

struct V { float x,y,z,w ; };
static const unsigned NUM_VPOS = 3 ; 

V apos[NUM_VPOS] = 
{
    { -0.1f , -0.1f,  0.f,  1.f }, 
    { -0.1f ,  0.1f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};

V bpos[NUM_VPOS] = 
{
    {  0.2f , -0.2f,  0.f,  1.f }, 
    {  0.2f ,  0.2f,  0.f,  1.f },
    {  0.f ,   0.f,   0.f,  1.f }
};


static const unsigned NUM_INST = 8 ; 
V ipos[NUM_INST] = 
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


V jpos[NUM_INST] = 
{
    {   0.1f ,   -0.1f,   0.f,  1.f }, 
    {   0.2f ,   -0.2f,   0.f,  1.f },
    {   0.3f ,   -0.3f,   0.f,  1.f },
    {   0.4f ,   -0.4f,   0.f,  1.f },
    {  -0.1f ,    0.1f,   0.f,  1.f }, 
    {  -0.2f ,    0.2f,   0.f,  1.f },
    {  -0.3f ,    0.3f,   0.f,  1.f },
    {  -0.4f ,    0.4f,   0.f,  1.f }
};



struct App 
{
    static const unsigned QSIZE  ; 
    static const unsigned LOC_vPosition ; 
    static const unsigned LOC_iPosition ; 
    static const char*    vertSrc ; 
    static const char*    fragSrc ; 

    Buf* a ; 
    Buf* b ; 
    Buf* i ; 
    Buf* j ; 

    GLuint ai ; 
    GLuint aj ; 
    GLuint bi ; 
    GLuint bj ; 

    App( Buf* a_, Buf* b_, Buf* i_, Buf* j_);
    void upload(Buf* buf);
    GLuint makeVertexArray( GLuint vbo, GLuint ibo );
};

const unsigned App::QSIZE = 4*sizeof(float) ; 
const unsigned App::LOC_vPosition = 0 ; 
const unsigned App::LOC_iPosition = 1 ; 

const char* App::vertSrc = R"glsl(

    #version 400 core
    layout(location = 0) in vec4 vPosition ;
    layout(location = 1) in vec4 iPosition ;
    void main()
    {
        //gl_Position = vec4( vPosition.x, vPosition.y, vPosition.z, 1.0 ) ;  
        gl_Position = vec4( vPosition.x + iPosition.x, vPosition.y + iPosition.y, vPosition.z + iPosition.z, 1.0 ) ;  
    }
)glsl";

const char* App::fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


App::App(  Buf* a_, Buf* b_, Buf* i_, Buf* j_ )
    :
    a(a_),
    b(b_),
    i(i_),
    j(j_)
{
    upload(a) ;
    upload(b) ;
    upload(i) ;
    upload(j) ;

    ai = makeVertexArray(a->id,i->id);
    aj = makeVertexArray(a->id,j->id);
    bi = makeVertexArray(b->id,i->id);
    bj = makeVertexArray(b->id,j->id);
}

void App::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint App::makeVertexArray( GLuint vbo, GLuint ibo )
{
    // (buffer,attribute) descriptions are captured into the VAO VertexArrayObject
    GLuint vertexArray ; 
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(LOC_vPosition);
    glVertexAttribPointer(LOC_vPosition, 4, GL_FLOAT, GL_FALSE, QSIZE, (const GLvoid*)0 );

    glBindBuffer(GL_ARRAY_BUFFER, ibo );
    glEnableVertexAttribArray(LOC_iPosition);
    glVertexAttribPointer(LOC_iPosition, 4, GL_FLOAT, GL_FALSE, QSIZE, (const GLvoid*)0 );
    glVertexAttribDivisor(LOC_iPosition, 1 );

    return vertexArray ;
}


int main()
{
    Frame frame ; 

    Prog draw(App::vertSrc, NULL, App::fragSrc ) ; 
    draw.compile();
    draw.create();

    glBindAttribLocation(draw.program, App::LOC_vPosition , "vPosition");
    glBindAttribLocation(draw.program, App::LOC_iPosition , "iPosition");

    draw.link();
    errcheck("main0");

    Buf a( sizeof(apos),apos ) ; 
    Buf b( sizeof(bpos),bpos ) ; 
    Buf i( sizeof(ipos),ipos ) ; 
    Buf j( sizeof(jpos),jpos ) ; 

    App app(&a, &b, &i, &j );
    unsigned count(0) ; 

    GLuint va = app.bj ; 

    while (!glfwWindowShouldClose(frame.window) && count++ < 100)
    {
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray( va );
        glDrawArraysInstanced(GL_TRIANGLES, 0, NUM_VPOS, NUM_INST );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    draw.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


