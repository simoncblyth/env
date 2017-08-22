#include "DEMO_API_EXPORT.hh"
#include <string>
#include <glm/glm.hpp>

struct DEMO_API Vue
{
    // model frame (-1:1 cube)  
    glm::vec4 eye ; 
    glm::vec4 look ; 
    glm::vec4 up ; 

    

    Vue();
    void home();

    void setEye(float x, float y, float z);
    void setLook(float x, float y, float z);
    void setUp(float x, float y, float z);

    void getTransforms(const glm::mat4& m2w, glm::mat4& world2camera, glm::mat4& camera2world, glm::vec4& gaze ); 
    std::string desc();

};


