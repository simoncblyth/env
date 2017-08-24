#include "DEMO_API_EXPORT.hh"
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct DEMO_API BB
{
    glm::vec3 min ; 
    glm::vec3 max ; 

    BB(float extent=0.f);
    static BB* FromVert(const std::vector<glm::vec4>& vert);  
    static BB* FromMat( const std::vector<glm::mat4>& mat );  

    std::string desc() const ;
    glm::vec4 get_center_extent() const  ;

    void include(const glm::vec3& p);
    bool is_empty() const ;
    void set_empty();
};
