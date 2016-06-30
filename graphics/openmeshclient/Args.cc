#include <iostream>
#include "Args.hh"
   
Args::Args(int argc_, char** argv_) 
     :
     argc(argc_),
     argv(argv_)
{
}
void Args::Summary(const char* msg)
{
    std::cerr << msg << std::endl ;
    for(int i=0 ; i < argc ; i++)
       std::cerr << argv[i] << std::endl ; 

}
