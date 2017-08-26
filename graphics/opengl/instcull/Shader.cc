
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "Shader.hh"


const unsigned Shader::LOC_VertexPosition = 0 ; 

const char* Shader::vertSrc = R"glsl(

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

const char* Shader::fragSrc = R"glsl(
    #version 400 core 
    out vec4 fColor ; 
    void main()
    {
        fColor = vec4(0.0, 0.0, 1.0, 1.0) ;  
    }
)glsl";


const unsigned Shader::QSIZE = sizeof(float)*4 ; 

Shader::Shader()
    :
    draw(new Prog(vertSrc, NULL, fragSrc )),
    uniform(new ShaderUniform)
{
    init();
    initUniformBuffer();
}

void Shader::destroy()
{
    draw->destroy();
}

void Shader::init()
{
    draw->compile();
    draw->create();

    glBindAttribLocation(draw->program, LOC_VertexPosition , "VertexPosition");

    draw->link();

    GLuint uniformBlockIndex = glGetUniformBlockIndex(draw->program, "MatrixBlock") ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(draw->program, uniformBlockIndex,  uniformBlockBinding );

    GU::errchk("Shader::init");
}

void Shader::initUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ShaderUniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);

    GU::errchk("Shader::initUniformBuffer");
}

void Shader::updateMVP( const glm::mat4& w2c)
{
    uniform->ModelViewProjection = w2c  ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(ShaderUniform), this->uniform);
}


GLuint Shader::createVertexArray(GLuint vertexBO, GLuint elementBO) 
{
    GLuint vloc =  LOC_VertexPosition ;

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBO);
    glEnableVertexAttribArray( vloc ); 
    glVertexAttribPointer( vloc , 4, GL_FLOAT, GL_FALSE, QSIZE, (void*)0);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBO);
    // https://stackoverflow.com/questions/8973690/vao-and-element-array-buffer-state

    GU::errchk("Shader::createVertexArray");

    return vertexArray;
}






