#include <stdio.h>
#include "MQ.h"

int main(int argc, char const * const *argv) {
   MQ::Create();
   gMQ->SendString("hello from notifymq_sendstring "); 
   return 0;
}
