#pragma once

#include "OMC_API_EXPORT.hh"

struct OMC_API Args {
   int    argc ; 
   char** argv ;
    
   Args(int argc_, char** argv_);
   void  Summary(const char* msg="Args::Summary") ;
};

