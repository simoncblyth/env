#include <iostream>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "G.hh"
#include "Cam.hh"


std::string Cam::desc() const 
{
    std::stringstream ss ; 
    glm::mat4 projection = getProjection() ;

    float left = getLeft() ; 
    float right = getRight() ; 
    float top = getTop() ; 
    float bottom = getBottom() ; 
    float aspect = getAspect() ; 

    glm::mat4 m(0.f) ; 

    m[0][0] = 2.f*near/(right-left) ;
    m[1][1] = 2.f*near/(top-bottom) ; 
    m[2][0] = (right+left)/(right-left) ;
    m[2][1] = (top+bottom)/(top-bottom) ; 
    m[2][2] = -(far+near)/(far-near) ;
    m[2][3] = -1.f ; 
    m[3][2] = -2.f*far*near/(far-near) ; 



    ss 
        << " basis " << basis << std::endl 
        << " factor " << factor << std::endl 
        << " near " << near << std::endl 
        << " far " << far << std::endl 
        << " zoom " << zoom << std::endl 
        << " aspect " << aspect << std::endl 
        << " parallel " << parallel << std::endl 
        << " left " << left << std::endl 
        << " right " << right << std::endl 
        << " bottom " << bottom << std::endl 
        << " top " << top << std::endl 
        << " yfov " << getYfov() 
        << std::endl 
        << G::gpresent("proj", projection )
        << std::endl 
        << G::gpresent("chek", m )
        << std::endl 
        ;


    return ss.str();
}


Cam::Cam(int width_, int height_, float basis )
    :
    zoom(1.f),
    parallel(false),
    parascale(1.f)
{
    setSize(width_, height_);
    setFocus(basis);
}

void Cam::setSize(int width_, int height_ )
{
    width = width_ ;
    height = height_ ;
}

void Cam::setFocus(float basis_, float factor_)
{
    basis = basis_ ; 
    factor = factor_ ; 
    near = basis/factor_ ; 
    far = basis*factor_ ; 
}


float Cam::getAspect() const { return (float)width/(float)height ; }  //  (> 1 for landscape) 
float Cam::getScale() const { return parallel ? parascale  : near ; }
float Cam::getTop() const {    return getScale() / zoom ; }
float Cam::getBottom() const { return -getScale() / zoom ; }
float Cam::getLeft() const {   return -getAspect() * getScale() / zoom ; }
float Cam::getRight() const {  return  getAspect() * getScale() / zoom ; }
float Cam::getNear() const { return near ; }
float Cam::getFar()  const { return far  ; }

void Cam::getFrustumVert(std::vector<glm::vec4>& vert)
{
    vert.push_back( { getLeft(),  getBottom(), getNear() , 1.f } );
    vert.push_back( { getRight(), getBottom(), getNear() , 1.f } );
    vert.push_back( { getRight(), getTop(),    getNear() , 1.f } );
    vert.push_back( { getLeft(),  getTop(),    getNear() , 1.f } );
    vert.push_back( { getLeft(),  getBottom(), getFar()  , 1.f } );
    vert.push_back( { getRight(), getBottom(), getFar()  , 1.f } );
    vert.push_back( { getRight(), getTop(),    getFar()  , 1.f } );
    vert.push_back( { getLeft(),  getTop(),    getFar()  , 1.f } );
}





void Cam::setYfov(float yfov_deg)
{
    // setYfov(90.) -> setZoom(1.)
    // fov = 2atan(1/zoom)
    // zoom = 1/tan(fov/2)

    zoom = 1.f/glm::tan(yfov_deg*0.5f*glm::pi<float>()/180.f );
}

float Cam::getYfov() const 
{
    float yfov_deg = 2.f*glm::atan(1.f/zoom)*180.f/glm::pi<float>() ;
    return yfov_deg ;
}

glm::mat4 Cam::getPerspective() const   // seems not used
{
    return glm::perspective(getYfov(), getAspect(), near, far);
}

glm::mat4 Cam::getProjection() const 
{
    return parallel ? getOrtho() : getFrustum() ;
}
glm::mat4 Cam::getOrtho() const 
{
    return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), near, far );
}
glm::mat4 Cam::getFrustum() const 
{ 
    return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), near, far );
}


const char* Cam::doc = R"doc(

Perspective Projection
-------------------------

Matrix from getFrustum() uses::

        glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() )

with inputs:

          left          -m_aspect*getScale()/m_zoom
          right          m_aspect*getScale()/m_zoom 
          top            getScale()/m_zoom
          bottom        -getScale()/m_zoom
          near 
          far

Note the opposing getScale() and zoom

The matrix is of below form, note the -1 which preps the perspective divide by z 
on converting from homogenous (see glm-)
 
     |   2n/(r-l)          0             0            0     | 
     |    0             2n/(t-b)         0            0     |
     |  (r+l)/(r-l)   (t+b)/(t-b)   -(f+n)/(f-n)     -1     |
     |    0               0         -2 f n/(f-n)      0     |

Summetric r+l = t+b = 0  r - l = a*h, t - b = h , scale = near , z = zoom

 
     |   z/a               0           0            0     | 
     |    0                z           0            0     |
     |    0                0        -(f+n)/(f-n)     -1     |
     |    0                0        -2 f n/(f-n)      0     |


Setting f = q*n   q > 1 

 -(f+n)/(f-n)  =  -n(q+1)/(n*(q-1) = -(q+1)/(q-1)    q large -> -1 

   -2f*n/(f-n) =   -2 q n^2 / n(q-1) = -2 n q /(q-1)    q large -> -2*n 


Large q 

     |   zoom/a            0           0            0     | 
     |    0             zoom            0            0     |
     |    0                0          -1           -1     |
     |    0                0        -2*n            0     |


 
     x' = x * zoom/aspect
     y' = y * zoom
     z' = -z - w 
     w' = -2*n*z 





/usr/local/opticks/externals/glm/glm-0.9.6.3/glm/gtc/matrix_transform.inl


::

    189     template <typename T>
    190     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> frustum
    191     (
    192         T left,
    193         T right,
    194         T bottom,
    195         T top,
    196         T nearVal,
    197         T farVal
    198     )
    199     {
    200         tmat4x4<T, defaultp> Result(0);
    201         Result[0][0] = (static_cast<T>(2) * nearVal) / (right - left);
    202         Result[1][1] = (static_cast<T>(2) * nearVal) / (top - bottom);
    203         Result[2][0] = (right + left) / (right - left);
    204         Result[2][1] = (top + bottom) / (top - bottom);
    205         Result[2][2] = -(farVal + nearVal) / (farVal - nearVal);
    206         Result[2][3] = static_cast<T>(-1);
    207         Result[3][2] = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);
    208         return Result;
    209     }



Normally this results in the FOV changing on changing the near distance.  
To avoid this getScale() returns m_near for perspective projection.  
This means the effective screen size scales with m_near, so FOV stays 
constant as m_near is varied, the only effect of changing m_near is 
to change the clipping of objects.

In order to change the FOV use the zoom setting.
 
  
Orthographic Projection
------------------------

Matrix from getOrtho() uses the below with same inputs as perspective::

      glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );

has form (see glm-):

    | 2/w  0   0        -(r+l)/(r-l)   |
    |  0  2/h  0        -(t+b)/(t-b)   |
    |  0   0  -2/(f-n)  -(f+n)/(f-n)   |
    |  0   0   0          1            |


189     template <typename T>
190     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> frustum
1
171     template <typename T>
172     GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> ortho
173     (
174         T left,
175         T right,
176         T bottom,
177         T top
178     )
179     {
180         tmat4x4<T, defaultp> Result(1);
181         Result[0][0] = static_cast<T>(2) / (right - left);
182         Result[1][1] = static_cast<T>(2) / (top - bottom);
183         Result[2][2] = - static_cast<T>(1);
184         Result[3][0] = - (right + left) / (right - left);
185         Result[3][1] = - (top + bottom) / (top - bottom);
186         return Result;
187     }
188 


)doc";







