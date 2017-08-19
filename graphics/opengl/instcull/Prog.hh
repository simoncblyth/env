#pragma once

struct Prog
{
    static const char* ShaderType(GLenum type);
    const char* vertSrc ;
    const char* geomSrc ;
    const char* fragSrc ;

    bool vert ; 
    bool geom ; 
    bool frag ;

    unsigned program ; 
    unsigned vertShader ;
    unsigned geomShader ;
    unsigned fragShader ;


    Prog(const char* vertSrc_, const char* geomSrc_,  const char* fragSrc_);

    void compile();
    unsigned compile(GLenum type, const char* src);
    void create();
    void link();
    void destroy();
    
    int getAttribLocation(const char* att) const ;
};




