#include <stdio.h>
#include "MQ.h"

int main(int argc, char const * const *argv) {
   MQ::Create();
   //const char* smry = gMQ->Summary(); 
   const char* smry = "exchange fanout.exchange exchangeType fanout queue grid1.phys.ntu.edu.tw routingKey default.routingkey passive 0 durable 0 autoDelete 1 exclusive 0 dayabaysoft@grid1.phys.ntu.edu.tw Fri Nov 27 18:27:00 2009";
   printf("sending string \"%s\" \n", smry );
   gMQ->SendString( smry ); 
   return 0;
}
