#include <stdio.h>
#include "MQ.h"

const int max = 256 ;
const char* other_key   = "other.routingkey" ;
const char* default_key = "default.routingkey" ;

int main(int argc, char const * const *argv) {
   MQ::Create();

   int n = 100 ;
   char smry[max];
   for(Int_t i = 0 ; i < n ; i++){
      const char* key = i % 2 == 0 ? default_key : other_key ;
      snprintf( smry, max, " %s %d %s  ", gMQ->Summary() , i , key   );
      gMQ->SendString( smry , key );
   } 
   return 0;
}
