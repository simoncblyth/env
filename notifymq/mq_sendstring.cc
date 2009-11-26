#include <stdio.h>
#include "MQ.h"

int main(int argc, char const * const *argv) {
   MQ::Create();
   const char* smry = gMQ->Summary(); 
   gMQ->SendString( smry ); 
   return 0;
}
