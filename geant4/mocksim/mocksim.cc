#include <stdio.h>  
#include <stdlib.h>    

#include "G4GDMLParser.hh"

int main(int argc, char** argv)
{
   const char* envkey = "DAE_NAME_DYB_GDML" ;
   const char* path = getenv(envkey);
   if(path == NULL){
      printf("envkey %s not in environment : use \"export-;export-export\" to define it \n", envkey);
      return 1;
   }

   G4GDMLParser fParser ; 
   bool validate = false ; 
   fParser.Read(path,validate);

   return 0 ; 
}


