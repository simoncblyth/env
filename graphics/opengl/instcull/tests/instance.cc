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
#include <glm/glm.hpp>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"



struct Uniform
{
    glm::mat4 ModelView ; 
    glm::mat4 ModelViewProjection ;

    Uniform() 
        :   
        ModelView(1.f), 
        ModelViewProjection(1.f)
    {}  ;

};



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

    Uniform* uniform ; 
    GLuint ubo ; 

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

    void updateUniform();
    void setupUniformBuffer();



};

const unsigned App::QSIZE = 4*sizeof(float) ; 
const unsigned App::LOC_vPosition = 0 ; 
const unsigned App::LOC_iPosition = 1 ; 

const char* App::vertSrc = R"glsl(

    #version 400 core

    uniform MatrixBlock
    {
        mat4 ModelView;
        mat4 ModelViewProjection;
    }  matrices  ;

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
    uniform(new Uniform),
    ubo(0),
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

    setupUniformBuffer();
}

void App::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void App::setupUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, this->ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(Uniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    std::cout << "sizeof(Uniform) " << sizeof(Uniform) << std::endl ; 


    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, ubo);
}

void App::updateUniform()
{
    uniform->ModelView = glm::mat4(1.) ;
    uniform->ModelViewProjection = glm::mat4(1.)  ;

    glBindBuffer(GL_UNIFORM_BUFFER, this->ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), this->uniform);
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

    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw.program, "matrices") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX);

    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw.program, uniformBlockIndex,  uniformBlockBinding );


    errcheck("main0");

    Buf a( NUM_VPOS, sizeof(apos),apos ) ; 
    Buf b( NUM_VPOS, sizeof(bpos),bpos ) ; 
    Buf i( NUM_INST, sizeof(ipos),ipos ) ; 
    Buf j( NUM_INST, sizeof(jpos),jpos ) ; 

    App app(&a, &b, &i, &j );
    unsigned count(0) ; 

    GLuint va = app.bj ; 

    while (!glfwWindowShouldClose(frame.window) && count++ < 100)
    {

    
        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        app.updateUniform(); 


        glBindVertexArray( va );
        glDrawArraysInstanced(GL_TRIANGLES, 0, NUM_VPOS, NUM_INST );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    draw.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


