

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GU.hh"
#include "Prog.hh"
#include "SContext.hh"
#include "InstShader.hh"


const char* SContext::uniformBlockName = "MatrixBlock" ; 
const char* SContext::uniformBlockSrc = R"glsl(

    // incorporated from SContext::uniformBlockSrc
    uniform MatrixBlock  
    {
        mat4 ModelViewProjection;
    } ;

)glsl";


const char* SContext::ReplaceUniformBlockToken(const char* vertSrc)  // static
{
    std::string vsrc(vertSrc); 
    GU::ReplaceAll(vsrc, "$UniformBlock", uniformBlockSrc );
    return strdup(vsrc.c_str()); 
}


SContext::SContext()
    :
    uniform(new SContextUniform)
{
    initUniformBuffer();
}



void SContext::initUniformBuffer()
{
     // same UBO can be used from all shaders
    glGenBuffers(1, &this->uniformBO);
    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(SContextUniform), this->uniform, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    GLuint binding_point_index = 0 ;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding_point_index, this->uniformBO);

    GU::errchk("SContext::initUniformBuffer");
}

void SContext::bindUniformBlock(GLuint program)
{
    GLuint uniformBlockIndex = glGetUniformBlockIndex(program,  uniformBlockName ) ;
    assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
    GLuint uniformBlockBinding = 0 ; 
    glUniformBlockBinding(program, uniformBlockIndex,  uniformBlockBinding );
}






void SContext::updateMVP( const glm::mat4& w2c)
{
    uniform->ModelViewProjection = w2c  ;  

    glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(SContextUniform), this->uniform);
}


