
#include <cstdio>
#include <cmath>  

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>

#include "NGLM.hpp"

#include "PLOG.hh"

// npy-
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

#include "Trackball.hh"


const char* Trackball::PREFIX = "trackball" ;
const char* Trackball::getPrefix()
{
   return PREFIX ; 
}


const char* Trackball::RADIUS = "radius" ;
const char* Trackball::ORIENTATION = "orientation" ;
const char* Trackball::TRANSLATE = "translate" ;
const char* Trackball::TRANSLATEFACTOR = "translatefactor" ;



std::vector<std::string> Trackball::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(RADIUS);
    tags.push_back(ORIENTATION);
    tags.push_back(TRANSLATE);
    tags.push_back(TRANSLATEFACTOR);
    return tags ; 
}








float Trackball::getTranslationMin()
{
    return -m_translate_max ; 
}

float Trackball::getTranslationMax()
{
    return m_translate_max ; 
}


float* Trackball::getRadiusPtr()
{
    return &m_radius ;
}
float Trackball::getRadiusMin()
{
    return m_radius_clip[0] ; 
}
float Trackball::getRadiusMax()
{
    return m_radius_clip[1] ; 
}



float* Trackball::getTFactorPtr()
{
    return &m_translatefactor ;
}
float Trackball::getTFactorMin()
{
    return m_translatefactor_clip[0] ; 
}
float Trackball::getTFactorMax()
{
    return m_translatefactor_clip[1] ; 
}







Trackball::Trackball() :
          m_radius(1.f),
          m_translatefactor(1000.f),
          m_drag_renorm(97),
          m_drag_count(0),
          m_changed(true)
{
    home();

    setTranslateMax(1e5);
    setRadiusClip(0.1f, 10.f);
    setTranslateFactorClip(10.f, 1e6);
}


void Trackball::setTranslateMax(float _max)
{
    m_translate_max = _max ; 
}

void Trackball::setRadiusClip(float _min, float _max)
{
    m_radius_clip[0] = _min ;  
    m_radius_clip[1] = _max ;  
}

void Trackball::setTranslateFactorClip(float _min, float _max)
{
    m_translatefactor_clip[0] = _min ;  
    m_translatefactor_clip[1] = _max ;  
}


float Trackball::getRadius()
{
    return m_radius ; 
}
float Trackball::getTranslateFactor()
{
    return m_translatefactor  ; 
}


void Trackball::setRadius(float r)
{
    m_radius = r ; 
    m_changed = true ; 
}
void Trackball::setTranslateFactor(float tf)
{
    m_translatefactor = tf ; 
    m_changed = true ; 
}
void Trackball::setTranslate(float x, float y, float z)
{
    m_translate.x = x;
    m_translate.y = y;
    m_translate.z = z;
    m_changed = true ; 
}
 
void Trackball::home()
{
    m_translate.x = 0.f ; 
    m_translate.y = 0.f ; 
    m_translate.z = 0.f ; 
    setOrientation(0.f, 0.f);
    m_changed = true ; 
}
void Trackball::zoom_to(float , float , float , float dy)
{
    m_translate.z += dy*m_translatefactor ;
    m_changed = true ; 
} 
void Trackball::pan_to(float , float , float dx, float dy)
{
    m_translate.x += dx*m_translatefactor ;
    m_translate.y += dy*m_translatefactor ;
    m_changed = true ; 
} 

bool Trackball::hasChanged()
{
    return m_changed ; 
}
void Trackball::setChanged(bool changed)
{
    m_changed = changed ; 
}























glm::vec3 Trackball::getTranslation()
{
    return m_translate;
}

float* Trackball::getTranslationPtr()
{
    return glm::value_ptr(m_translate);
}

float* Trackball::getOrientationPtr()
{
    return glm::value_ptr(m_orientation);
}

std::string Trackball::getOrientationString()
{
    return gformat(m_orientation) ;
}



glm::mat4 Trackball::getTranslationMatrix()
{
    return glm::translate(  glm::mat4(1.0f), m_translate);
}

glm::mat4 Trackball::getCombinedMatrix()
{
    // http://stackoverflow.com/questions/9920624/glm-combine-rotation-and-translation
    //
    // glm::translate,  translates before rotates
    //
    //        rot * trans;
    //

    return glm::translate( getOrientationMatrix(), m_translate);
}

void Trackball::getCombinedMatrices(glm::mat4& rt, glm::mat4& rti)
{
    glm::mat4 rot  = getOrientationMatrix();  
    glm::mat4 irot = glm::transpose(rot);

    glm::mat4 tra(glm::translate(m_translate));  
    glm::mat4 itra(glm::translate(-m_translate));  

    rt = rot * tra  ;    // translates then rotate

    rti = itra * irot ;  //    

    //  rti * rt = itra * irot * rot * tra = identity 
    //

} 

void Trackball::getOrientationMatrices(glm::mat4& rot, glm::mat4& irot)
{
    rot  = getOrientationMatrix();  
    irot = glm::transpose(rot);
} 

void Trackball::getTranslationMatrices(glm::mat4& tra, glm::mat4& itra)
{
    tra = glm::translate(m_translate);  
    itra = glm::translate(-m_translate);  
} 



void Trackball::configureI(const char* , std::vector<int> )
{
}



void Trackball::configureF(const char* name, std::vector<float> values)
{
     if(values.empty())
     {
         printf("Trackball::configureF %s no values \n", name);
     }
     else         
     {
         float vlast = values.back() ;

#ifdef VERBOSE
         LOG(info) << "Trackball::configureF"
                   << " name " << name 
                   << " vals " << values.size()
                   ;

         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );
#endif

         if(      strcmp(name, RADIUS) ==  0)          setRadius(vlast);
         else if( strcmp(name, TRANSLATEFACTOR) ==  0) setTranslateFactor(vlast);
         else
              printf("Trackball::configureF ignoring unknown parameter %s : %10.3f \n", name, vlast); 
     }
}
 

void Trackball::configureS(const char* name, std::vector<std::string> values)
{
     if(values.empty())
     {
         printf("Trackball::configureS %s no values \n", name);
     }
     else         
     {
         std::string  vlast = values.back() ;
#ifdef VERBOSE
         LOG(info) << "Trackball::configureS"
                   << " name " << name 
                   << " vals " << values.size()
                   ;

         for(size_t i=0 ; i < values.size() ; i++ ) printf("%20s ", values[i].c_str());
         printf(" : vlast %20s \n", vlast.c_str() );
#endif
         set(name, vlast);
    }
}



bool Trackball::accepts(const char* name)
{
    return 
         strcmp(name,TRANSLATE)==0  ||
         strcmp(name,TRANSLATEFACTOR)==0  ||
         strcmp(name,ORIENTATION)==0 ||
         strcmp(name,RADIUS)==0 ;
}

void Trackball::configure(const char* name, const char* value_)
{
    std::string value(value_);
    set(name, value);
}

void Trackball::set(const char* name, std::string& s)
{
    if(      strcmp(name, TRANSLATE)   ==  0)     setTranslate(s);
    else if( strcmp(name, ORIENTATION) ==  0)     setOrientation(s);
    else if( strcmp(name, RADIUS) ==  0)          setRadius(s);
    else if( strcmp(name, TRANSLATEFACTOR) ==  0) setTranslateFactor(s);
    else
        printf("Trackball::set ignoring unknown parameter %s : %s \n", name, s.c_str()); 
}


std::string Trackball::get(const char* name)
{
    std::string s ; 
    if(strcmp(name,TRANSLATE)==0)
    {
         glm::vec3 v = getTranslation(); 
         s = gformat(v);
    }
    else if( strcmp(name, ORIENTATION) ==  0)
    {
         glm::quat q = getOrientation();
         s = gformat(q);
    }
    else if( strcmp(name, RADIUS) ==  0)
    {
         float r = getRadius();
         s = gformat(r);
    }
    else if( strcmp(name, TRANSLATEFACTOR) ==  0)
    {
         float f = getTranslateFactor();
         s = gformat(f);
    }
    else
         printf("Trackball::get bad name %s\n", name);

    return s ;
}


void Trackball::setRadius(std::string s)
{
    setRadius(gfloat_(s)); 
}

void Trackball::setTranslateFactor(std::string s)
{
    setTranslateFactor(gfloat_(s)); 
}

 
void Trackball::setOrientation(std::string _tp)
{
    std::vector<std::string> tp;
    boost::split(tp, _tp, boost::is_any_of(","));

    if(tp.size() == 2 )
    {
        float theta = boost::lexical_cast<float>(tp[0]); 
        float phi   = boost::lexical_cast<float>(tp[1]); 
        setOrientation(theta,phi);
    }
    else if(tp.size() == 4 )
    {
        glm::quat q = gquat(_tp);
        setOrientation(q);  
    }
    else
    {
        printf("Trackball::setOrientation malformed _tp : %s \n", _tp.c_str() );
    }
}

void Trackball::setTranslate(std::string _xyz)
{
    glm::vec3 v = gvec3(_xyz);
    setTranslate(v.x,v.y,v.z);
}


void Trackball::setOrientation(float _theta, float _phi)
{
    m_theta_deg = _theta ; 
    m_phi_deg  = _phi ;  

    setOrientation();
}

void Trackball::setOrientation()
{
    float pi = boost::math::constants::pi<float>() ;
    float theta = m_theta_deg*pi/180.f ;
    float phi   = m_phi_deg*pi/180.f ;

    glm::quat xrot(cos(0.5f*theta),sin(0.5f*theta),0.f,0.f);
    glm::quat zrot(cos(0.5f*phi),  0.f,0.f,sin(0.5f*phi));
    glm::quat q = xrot * zrot ; 
    setOrientation(q);
}


glm::quat Trackball::getOrientation()
{
   return m_orientation ;  
}

glm::mat4 Trackball::getOrientationMatrix()
{
   return glm::toMat4(m_orientation) ;  
}

void Trackball::setOrientation(glm::quat& q)
{
    m_orientation = q  ; 
    m_changed = true ; 
}

void Trackball::drag_to(float x, float y, float dx, float dy)
{
    //printf("Trackball::drag_to %10.3f %10.3f %10.3f %10.3f \n", x, y, dx, dy);

    m_drag_count += 1 ; 

    glm::quat drag = rotate(x,y,dx,dy);

    bool bad =
                std::isnan(drag.x) || 
                std::isnan(drag.y) || 
                std::isnan(drag.z) || 
                std::isnan(drag.w) ||
                std::isinf(drag.x) || 
                std::isinf(drag.y) || 
                std::isinf(drag.z) || 
                std::isinf(drag.w)  ;

    if(bad)
    {
        LOG(warning) << "Trackball::drag_to IGNORING bad drag " ; 
        return ;  
    }

    glm::quat qrot = m_orientation * drag ;   // perturb orientation by drag rotation  

    if(m_drag_count > m_drag_renorm)
    {
        //print(qrot, "Trackball::drag_to before renorm");
        qrot = glm::normalize(qrot);         
        //print(qrot, "Trackball::drag_to after renorm");
        m_drag_count = 0 ;
    }

    setOrientation(qrot);
}


void Trackball::Summary(const char* msg)
{
    printf(" trackballradius  %10.3f \n", m_radius );
    print(getOrientation(), msg);
    //print(getOrientationMatrix(), msg);
}


glm::quat Trackball::rotate(float x, float y, float dx, float dy)
{
    // p0, p1 are positions of screen before and after coordinates 
    // projected onto a deformed virtual trackball

    glm::vec3 p0(x   ,    y, project(m_radius,x,y)); 
    glm::vec3 p1(x+dx, y+dy, project(m_radius,x+dx,y+dy)); 
   
    // axis of rotation        
    glm::vec3 axis = glm::cross(p1, p0);

    // angle of rotation
    float t = glm::clamp(glm::length(p1-p0)/(2.f*m_radius), -1.f, 1.f ) ;
    float phi = 2.f * asin(t) ;

    glm::quat q = glm::angleAxis( phi, glm::normalize(axis) );

#ifdef DEBUG
    print(p0,  "Trackball::rotate p0");
    print(p1,  "Trackball::rotate p1");
    print(axis,"Trackball::rotate axis");
    printf("Trackball::rotate t %15.5f phi %15.5f \n", t, phi );
    print(q,   "Trackball::rotate q");
#endif

    return q ; 
}
 



float Trackball::project(float r, float x, float y)
{
  /*
        Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
        if we are away from the center of the sphere.

        For points inside xy circle::

                  d^2 = x^2 + y^2

                    d < r / sqrt(2)   

                  d^2 < r^2 / 2 

            x^2 + y^2 < r^2 / 2 
   

        determine z from::

                z^2 + d^2 = r^2 

        So are projecting onto sphere the center of which is on the screen plane.
  */      
 
  float z, t, d ;
  d = sqrt(x*x + y*y);
  if (d < r * 0.70710678118654752440f)
  {     
      z = sqrt(r*r - d*d);
  }
  else
  {   
     t = r / 1.41421356237309504880f ;
     z = t*t / d  ;
  }
  return z;
}


