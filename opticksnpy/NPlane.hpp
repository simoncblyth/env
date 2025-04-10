#pragma once

#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API nbbox {

    // NO CTOR

    void dump(const char* msg);

    nvec4 min ; 
    nvec4 max ; 
};


// "ctor" assuming rotational symmetry around z axis
inline NPY_API nbbox make_nbbox(float zmin, float zmax, float ymin, float ymax)
{
    nbbox bb ; 
    bb.min = make_nvec4( ymin, ymin, zmin, 0) ;
    bb.max = make_nvec4( ymax, ymax, zmax, 0) ;
    return bb ;
}

inline NPY_API nbbox make_nbbox()
{
    return make_nbbox(0,0,0,0) ;
}



struct NPY_API nplane {
    // http://mathworld.wolfram.com/Plane.html
    // xyz: normalized normal vector, w:distance from origin

    void dump(const char* msg);

    nvec4 param ; 
};


inline NPY_API nplane make_nplane(float x, float y, float z, float w)
{  
   nplane pl ; pl.param.x = x ; pl.param.y = y ; pl.param.z = z ; pl.param.w = w ; return pl ; 
}

inline NPY_API nplane make_nplane(const nvec4& p)
{  
   nplane pl ; pl.param.x = p.x ; pl.param.y = p.y ; pl.param.z = p.z ; pl.param.w = p.w ; return pl ; 
}


struct NPY_API ndisc {
    float z() const;
    nplane plane ;
    float radius ;  

    void dump(const char* msg);
};


inline NPY_API ndisc make_ndisc(const nplane& plane_, float radius_) 
{
   ndisc d ; d.plane = plane_ ; d.radius = radius_ ; return d ; 
}




