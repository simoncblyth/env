// name=switch_nested_if_break.cc && cat $name && gcc $name && RED=0 GREEN=1 BLUE=1 CYAN=0 MAGENTA=1 YELLOW=1 ./a.out ; echo $?

#include <string>
#include <iostream>
#include <stdlib.h>   

int getenvvar(const char* name, int def)
{
   int ivar = def ; 
   char* evar = getenv(name);
   if (evar!=NULL) ivar = atoi(evar);
   return ivar ;
}


int main(int argc,char** argv)
{
   int red     = getenvvar("RED",0) ;
   int green   = getenvvar("GREEN",0) ; 
   int blue    = getenvvar("BLUE",0) ; 
   int cyan    = getenvvar("CYAN",0) ; 
   int magenta = getenvvar("MAGENTA",0) ; 
   int yellow  = getenvvar("YELLOW",0) ; 

   int rc = 0 ; 
   int stage = 0 ; 
 

   /* ANTI-PATTERN AS FRAGILE : VERY BREAK-ABLE */
 
   switch(stage)
   {
   case 0:
           if(red){
                if(cyan){
                   rc = 1 ; 
                } else if(magenta){
                   rc = 2 ; 
                   break ; 
                } else if(yellow){
                   rc = 3 ; 
                } 
                //break ; 
           } else {
              rc = 30 ; 
           }
           rc = 40 ; 
           break ; 
   } 
   return rc ;
}




