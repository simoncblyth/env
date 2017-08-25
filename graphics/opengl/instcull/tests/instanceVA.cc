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
#include <glm/gtx/transform.hpp>

#include "Prog.hh"
#include "Frame.hh"
#include "Buf.hh"
#include "Pos.hh"
#include "GU.hh"


#define WITH_UBO 1

#ifdef WITH_UBO
struct Uniform
{
    glm::mat4 ModelView ; 
    glm::mat4 ModelViewProjection ;

    Uniform() 
        :   
        ModelView(0.1f), 
        ModelViewProjection(0.1f)
    {}  ;

};

#endif


struct App 
{
    static const unsigned QSIZE  ; 
    static const unsigned LOC_vPosition ; 
    static const unsigned LOC_iPosition ; 
    static const char*    vertSrc ; 
    static const char*    fragSrc ; 

#ifdef WITH_UBO
    Uniform* uniform ; 
    GLuint ubo ; 

    void updateUniform(float t);
    void setupUniformBuffer();
#endif

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


// seems preprocessor macros not woking inside raw blocks
// tokenization happens before preprocessor
// https://stackoverflow.com/questions/30997129/in-c11-what-should-happen-first-raw-string-expansion-or-macros

#ifdef WITH_UBO
const char* App::vertSrc = R"glsl(

    #version 400 core

    uniform MatrixBlock
    {
        mat4 ModelView;
        mat4 ModelViewProjection;
    } ;

    layout(location = 0) in vec4 vPosition ;
    layout(location = 1) in vec4 iPosition ;
    void main()
    {
        vec4 pos = vec4( vPosition.x + iPosition.x, vPosition.y + iPosition.y, vPosition.z + iPosition.z, 1.0 ) ;  
        gl_Position = ModelViewProjection * pos  ;

        //gl_Position = vec4( vPosition.x, vPosition.y, vPosition.z, 1.0 ) ;  
        //gl_Position = vec4( vPosition.x + iPosition.x, vPosition.y + iPosition.y, vPosition.z + iPosition.z, 1.0 ) ;  
    }
)glsl";

#else
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

#endif



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
#ifdef WITH_UBO
    uniform(new Uniform),
    ubo(0),
#endif
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

#ifdef WITH_UBO
    setupUniformBuffer();
#endif
}

void App::upload(Buf* buf)
{
    glGenBuffers(1, &buf->id);
    glBindBuffer(GL_ARRAY_BUFFER, buf->id);
    glBufferData(GL_ARRAY_BUFFER, buf->num_bytes, buf->ptr,  GL_STATIC_DRAW );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

#ifdef WITH_UBO
void App::setupUniformBuffer()
{
    // same UBO can be used from all shaders
    glGenBuffers(1, &this->ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, this->ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(Uniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    unsigned size = sizeof(Uniform) ;
    unsigned x_size = sizeof(float)*4*4*2 ; 

    std::cout 
         << "App::setupUniformBuffer"
         << " sizeof(Uniform) " << size 
         << " x_size " << x_size
         << std::endl 
         ; 

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, ubo);
}

void App::updateUniform(float t)
{
    glm::vec3 tla(-0.25f,-0.25f,0.f);
    tla *= t ; 

    glm::mat4 m = glm::translate(glm::mat4(1.f), tla);

    uniform->ModelView = m ;
    uniform->ModelViewProjection = m  ;

    glBindBuffer(GL_UNIFORM_BUFFER, this->ubo);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Uniform), this->uniform);
}
#endif


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


int main(int, char** argv)
{
    Frame frame(argv[0]) ; 

    Prog draw(App::vertSrc, NULL, App::fragSrc ) ; 
    draw.compile();
    draw.create();

    glBindAttribLocation(draw.program, App::LOC_vPosition , "vPosition");
    glBindAttribLocation(draw.program, App::LOC_iPosition , "iPosition");

    draw.link();


    GU::errchk("main0");

#ifdef WITH_UBO
    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw.program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform in shader otherwise it is optimized away and this will fail");

    std::cout 
         << " uniformBlockIndex " << uniformBlockIndex
         << " GL_INVALID_INDEX " << GL_INVALID_INDEX
         << std::endl
         ;

    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw.program, uniformBlockIndex,  uniformBlockBinding );
#endif


    Buf* a = Pos::a();
    Buf* b = Pos::b();
    Buf* i = Pos::i();
    Buf* j = Pos::j();

    App app(a, b, i, j );
    unsigned count(0) ; 

    GLuint va = app.bj ; 

    while (!glfwWindowShouldClose(frame.window) && count++ < 100)
    {

        int width, height;
        glfwGetFramebufferSize(frame.window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        double t = glfwGetTime();

#ifdef WITH_UBO
        app.updateUniform((float)t); 
#endif

        glBindVertexArray( va );
        glDrawArraysInstanced(GL_TRIANGLES, 0, a->num_items, i->num_items );

        glfwSwapBuffers(frame.window);
        glfwPollEvents();
    }

    draw.destroy();
    frame.destroy();

    exit(EXIT_SUCCESS);
}


