#pragma once

#include "OGLRAP_API_EXPORT.hh"

class OGLRAP_API Touchable {
    public:
        virtual ~Touchable(){}
        virtual unsigned int touch(int ix, int iy) = 0 ;
  
};
