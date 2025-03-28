#pragma once

#include "NPY_API_EXPORT.hh"

struct NPY_API NSlice {

     unsigned int low ; 
     unsigned int high ; 
     unsigned int step ; 
     const char*  _description ; 

     NSlice(const char* slice, const char* delim=":");
     NSlice(unsigned int low, unsigned int high, unsigned int step=1);

     const char* description();
     unsigned int count();
};

