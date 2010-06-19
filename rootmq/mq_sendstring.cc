#include <stdio.h>
#include "MQ.h"

const int max = 256 ;
const char* other_key   = "abt.test.other" ;
const char* default_key = "abt.test.string" ;

int main(int argc, char const * const *argv) {
   MQ::Create();
   int n = 10 ;
   char smry[max];
   for(Int_t i = 0 ; i < n ; i++){
      const char* key = i % 2 == 0 ? default_key : other_key ;
      snprintf( smry, max, " mq_sendstring i:%d key:%s  ", i , key   );
      gMQ->SendString( smry , key );
   } 
   
   return 0;
}
