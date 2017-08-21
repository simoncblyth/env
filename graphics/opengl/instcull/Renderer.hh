#pragma once

#include <vector>
struct Buf ; 

#include "DEMO_API_EXPORT.hh"


struct DEMO_API  Renderer
{
    unsigned vao;
    std::vector<Buf*> buffers ; 

    Renderer();

    void upload(Buf* buf, GLenum target, GLenum usage);
    void destroy();
};


