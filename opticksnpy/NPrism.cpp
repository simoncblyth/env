#include <cmath>
#include <cassert>
#include <cmath>

#include <boost/math/constants/constants.hpp>

#include "NPrism.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"


nprism::nprism(float apex_angle_degrees, float height_mm, float depth_mm, float fallback_mm)
{
    param.x = apex_angle_degrees  ;
    param.y = height_mm  ;
    param.z = depth_mm  ;
    param.w = fallback_mm  ;
}

nprism::nprism(const nvec4& param_)
{
    param = param_ ;
}

float nprism::height()
{
    return param.y > 0.f ? param.y : param.w ; 
}
float nprism::depth()
{
    return param.z > 0.f ? param.z : param.w ; 
}
float nprism::hwidth()
{
    float pi = boost::math::constants::pi<float>() ;
    return height()*tan((pi/180.f)*param.x/2.0f) ;
}


void nprism::dump(const char* msg)
{
    param.dump(msg);
}

npart nprism::part()
{
    // hmm more dupe of hemi-pmt.cu/make_prism
    // but if could somehow make vector types appear 
    // the same could use same code with CUDA ?

    float h  = height();
    float hw = hwidth();
    float d  = depth();

    nbbox bb ;
    bb.min = {-hw,0.f,-d/2.f, 0.f } ;
    bb.max = { hw,  h, d/2.f, 0.f } ;

    npart p ; 
    p.zero();            

    p.setParam(param) ; 
    p.setTypeCode(PRISM); 
    p.setBBox(bb);

    return p ; 
}


