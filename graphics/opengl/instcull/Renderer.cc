
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "Renderer.hh"


const unsigned Renderer::LOC_VertexPosition = 0 ; 

const char* Renderer::vertSrc = R"glsl(

    #version 400 core
    uniform MatrixBlock  
    {
        mat4 ModelViewProjection;
    } ;

    layout (location = 0) in vec4 VertexPosition;

    void main()
    {
        gl_Position = ModelViewProjection * VertexPosition ;
    }

)glsl";

const char* Renderer::fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


const unsigned Renderer::QSIZE = sizeof(float)*4 ; 

Renderer::Renderer()
    :
    draw(new Prog(vertSrc, NULL, fragSrc )),
    uniform(new RendererUniform)
{
    init();
    initUniformBuffer();
}

void Renderer::destroy()
{
    draw->destroy();
}

void Renderer::init()
{
    draw->compile();
    draw->create();

    glBindAttribLocation(draw->program, LOC_VertexPosition , "VertexPosition");

    draw->link();

    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw->program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw->program, uniformBlockIndex,  uniformBlockBinding );

    GU::errchk("Renderer::init");
}

void Renderer::initUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(RendererUniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);

    GU::errchk("Renderer::initUniformBuffer");
}

void Renderer::updateMVP( const glm::mat4& w2c)
{
    uniform->ModelViewProjection = w2c  ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(RendererUniform), this->uniform);
}


GLuint Renderer::createVertexArray(GLuint vertexBO) 
{
    GLuint vloc =  LOC_VertexPosition ;

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBO);
    glEnableVertexAttribArray( vloc ); 
    glVertexAttribPointer( vloc , 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);

    GU::errchk("Renderer::createVertexArray");

    return vertexArray;
}





