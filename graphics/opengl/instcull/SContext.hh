#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>

struct DEMO_API SContextUniform
{  
    glm::mat4 ModelViewProjection ;
};

struct DEMO_API SContext
{
    static const char* uniformBlockName ; 
    static const char* uniformBlockSrc ;
    static const char* ReplaceUniformBlockToken(const char* vertSrc);


    SContextUniform*   uniform ; 
    GLuint             uniformBO ; 

   
    SContext();
    void initUniformBuffer();
    void bindUniformBlock(GLuint program);

    void updateMVP( const glm::mat4& w2c);


};
