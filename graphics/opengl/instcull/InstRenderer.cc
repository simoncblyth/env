
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "InstRenderer.hh"


const unsigned InstRenderer::LOC_VertexPosition = 0 ; 
const unsigned InstRenderer::LOC_InstanceTransform = 1 ; 

const char* InstRenderer::vertSrc = R"glsl(

    #version 400 core
    uniform MatrixBlock  
    {
        mat4 ModelViewProjection;
    } ;

    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 InstanceTransform ;

    void main()
    {
        gl_Position = ModelViewProjection * InstanceTransform * VertexPosition ;
    }

)glsl";

const char* InstRenderer::fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


const unsigned InstRenderer::QSIZE = sizeof(float)*4 ; 

InstRenderer::InstRenderer()
    :
    draw(new Prog(vertSrc, NULL, fragSrc )),
    uniform(new InstRendererUniform)
{
    init();
    initUniformBuffer();
}

void InstRenderer::destroy()
{
    draw->destroy();
}

void InstRenderer::init()
{
    draw->compile();
    draw->create();

    glBindAttribLocation(draw->program, LOC_VertexPosition , "VertexPosition");
    glBindAttribLocation(draw->program, LOC_InstanceTransform , "InstanceTransform");

    draw->link();

    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw->program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw->program, uniformBlockIndex,  uniformBlockBinding );

    GU::errchk("InstRenderer::init");
}

void InstRenderer::initUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(InstRendererUniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);

    GU::errchk("InstRenderer::initUniformBuffer");
}

void InstRenderer::updateMVP( const glm::mat4& w2c)
{
    uniform->ModelViewProjection = w2c  ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(InstRendererUniform), this->uniform);
}


GLuint InstRenderer::createVertexArray(GLuint instanceBO, GLuint vertexBO) 
{
    GLuint vloc =  LOC_VertexPosition ;
    GLuint iloc =  LOC_InstanceTransform ;

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBO);
    glEnableVertexAttribArray( vloc ); 
    glVertexAttribPointer( vloc , 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);

    GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on top of each other
    glBindBuffer(GL_ARRAY_BUFFER, instanceBO);

    glEnableVertexAttribArray(iloc + 0);
    glVertexAttribPointer(    iloc + 0, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(0*QSIZE) );
    glVertexAttribDivisor(    iloc + 0, divisor );

    glEnableVertexAttribArray(iloc + 1);
    glVertexAttribPointer(    iloc + 1, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(1*QSIZE) );
    glVertexAttribDivisor(    iloc + 1, divisor );

    glEnableVertexAttribArray(iloc + 2);
    glVertexAttribPointer(    iloc + 2, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(2*QSIZE) );
    glVertexAttribDivisor(    iloc + 2, divisor );

    glEnableVertexAttribArray(iloc + 3);
    glVertexAttribPointer(    iloc + 3, 4, GL_FLOAT, GL_FALSE, 4*QSIZE, (void*)(3*QSIZE) );
    glVertexAttribDivisor(    iloc + 3, divisor );

    GU::errchk("InstRenderer::createVertexArray");

    return vertexArray;
}





