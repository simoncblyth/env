#pragma once
#include "stdlib.h"

class npyBuffer {
   public:
       npyBuffer(char* bytes, size_t size);
       virtual ~npyBuffer();
   private:
       char* m_bytes ;
       size_t m_size ; 
};



