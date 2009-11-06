#include <stdio.h>
#include "notifymq.h"

int main(int argc, char const * const *argv) {
   if (argc < 4) {
      fprintf(stderr, "Usage: notifymq_sendstring exchange routingkey messagebody\n");
      return 1;
   }
   notifymq_init();
   char const* exchange = argv[1];
   char const* routingkey = argv[2];
   char const* messagebody = argv[3];
   notifymq_sendstring( exchange , routingkey , messagebody );
   notifymq_cleanup();
   return 0;
}
