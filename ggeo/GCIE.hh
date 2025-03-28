#pragma once

#include <glm/fwd.hpp>

template <typename T> class GDomain ;
template <typename T> class GAry ;

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GCIE {
   public:
        static const char* sRGB_D65 ;
   public:
        GCIE(float wmin, float wmax, float wstep);
        void dump(const char* msg="GCIE::dump");
   public:
        glm::vec3 getXYZ(float wavelength); 
        glm::vec3 getRGB(float wavelength); 

        float getInterpolatedX(float wavelength);
        float getInterpolatedY(float wavelength);
        float getInterpolatedZ(float wavelength);

        float getXMax();
        float getYMax();
        float getZMax();
   private:
        void init();
   private:
       glm::mat3       m_XYZ2RGB ; 
       GDomain<float>* m_domain ; 
       GAry<float>*    m_nm ; 
       GAry<float>*    m_X ; 
       GAry<float>*    m_Y ; 
       GAry<float>*    m_Z ; 

};

#include "GGEO_TAIL.hh"


