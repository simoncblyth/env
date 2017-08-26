#pragma once

#include "DEMO_API_EXPORT.hh"
#include <glm/glm.hpp>


struct SContextUniform ;

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

    void update( const glm::mat4& world2clip, const glm::mat4& world2eye);

};
