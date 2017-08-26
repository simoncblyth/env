#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include "DEMO_API_EXPORT.hh"

struct DEMO_API Buf
{
    unsigned id ; 
    unsigned num_items ;
    unsigned num_bytes ;
    void*    ptr ;

    unsigned item_bytes() const ;

    Buf(unsigned num_items_, unsigned num_bytes_, void* ptr_) ;
    Buf* cloneEmpty() const ;
    
    void upload(GLenum target, GLenum usage );
    void uploadNull(GLenum target, GLenum usage );

    std::string desc() const ;
    void dump(const char* msg) const ; 

    static Buf* Make(const std::vector<glm::mat4>& mat) ;
    static Buf* Make(const std::vector<glm::vec4>& vert) ;
    static Buf* Make(const std::vector<unsigned>&  elem) ;

};



