#pragma once

#include "DEMO_API_EXPORT.hh"

#include <string>
#include <glm/glm.hpp>

struct DEMO_API G
{
    static std::string gpresent(const char* label, const glm::mat4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=20, unsigned mwid=5, bool flip=false );
    static std::string gpresent(const char* label, const glm::vec4& m, unsigned prec=3, unsigned wid=7, unsigned lwid=20, unsigned mwid=5);
    static std::string gpresent(const char* label, const glm::vec3& m, unsigned prec=3, unsigned wid=7, unsigned lwid=20, unsigned mwid=5);


};
