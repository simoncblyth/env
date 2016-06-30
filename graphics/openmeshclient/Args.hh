#pragma once

struct Args {
   int    argc ; 
   char** argv ;
    
   Args(int argc_, char** argv_);
   void  Summary(const char* msg="Args::Summary") ;
};

