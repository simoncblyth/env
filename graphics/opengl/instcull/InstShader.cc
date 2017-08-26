
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "SContext.hh"
#include "InstShader.hh"


const unsigned InstShader::LOC_VertexPosition = 0 ; 
const unsigned InstShader::LOC_VizInstanceTransform = 1 ; 

const char* InstShader::vertSrc = R"glsl(

    #version 400 core
    // InstShader::vertSrc

    $UniformBlock 

    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;

    void main()
    {
        gl_Position = ModelViewProjection * VizInstanceTransform * VertexPosition ;
    }

)glsl";

const char* InstShader::fragSrc = R"glsl(
    #version 400 core 
    // InstShader::fragSrc
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


const unsigned InstShader::QSIZE = sizeof(float)*4 ; 

InstShader::InstShader(SContext* context_)
    :
    context(context_),
    prog(new Prog(SContext::ReplaceUniformBlockToken(vertSrc), NULL, fragSrc ))
{
    initProgram();
    //initUniformBuffer();
}

void InstShader::destroy()
{
    prog->destroy();
}

void InstShader::initProgram()
{   
    prog->compile();
    prog->create();

    glBindAttribLocation(prog->program, LOC_VertexPosition , "VertexPosition");
    glBindAttribLocation(prog->program, LOC_VizInstanceTransform , "VizInstanceTransform");

    prog->link();

    context->bindUniformBlock(prog->program);

    GU::errchk("InstShader::init");
}


GLuint InstShader::createVertexArray(GLuint instanceBO, GLuint vertexBO, GLuint elementBO ) 
{
    GLuint vloc =  LOC_VertexPosition ;
    GLuint iloc =  LOC_VizInstanceTransform ;

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBO ); 

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

    GU::errchk("InstShader::createVertexArray");

    return vertexArray;
}


