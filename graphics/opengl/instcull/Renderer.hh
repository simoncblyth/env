#pragma once

#include <vector>
struct Buf ; 

struct Renderer
{
    unsigned vao;
    std::vector<Buf*> buffers ; 

    Renderer();

    void upload(Buf* buf, GLenum target, GLenum usage);
    void destroy();
};


