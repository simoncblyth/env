#include <stdio.h>
#include "MQ.h"


const int max = 256 ;


int main(int argc, char const * const *argv) {
   MQ::Create();

   char smry[max];
   for(Int_t i = 0 ; i < 100 ; i++){
      snprintf( smry, max, " %s %d ", gMQ->Summary() , i  );
      gMQ->SendString( smry , "default.routingkey" );
   } 
   return 0;
}
