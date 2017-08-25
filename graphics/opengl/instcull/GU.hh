#pragma once

#include "DEMO_API_EXPORT.hh"

#include <string>

struct DEMO_API GU
{
   static void errchk(const char* msg);
   static void ReplaceAll(std::string& subject, const char* search, const char* replace) ;

};
