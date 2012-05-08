/*

  Investigating conditions that yield the runtime error:

      terminate called after throwing an instance of 'std::logic_error'
       what():  basic_string::_S_construct NULL not valid

*/
#include <string>
#include <iostream>

const char* GetNULL()
{
   return NULL ;
}

int main(int argc,char** argv)
{

   // YES
   //std::string s(NULL) ;
   
   // NO   ... thinking temporary strings 
   //const char* c = NULL ;
   //std::cout << c << std::endl ;

   // NO
   //std::cout << NULL << std::endl ;

   // NO
   //std::cout << GetNULL() << std::endl ;
   

   return 0 ;
}

