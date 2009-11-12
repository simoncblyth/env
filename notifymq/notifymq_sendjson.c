#include <stdio.h>
#include <stdlib.h>
#include "notifymq.h"
#include "cJSON.h"


int main(int argc, char const * const *argv) {
   if (argc < 3) {
      fprintf(stderr, "Usage: notifymq_sendjson exchange routingkey \n");
      return 1;
   }

   int rc ;
   if((rc = notifymq_init())){
      fprintf(stderr, "ABORT: notifymq_init failed rc : %d \n", rc );
      return rc ;
   }
   char const* exchange = argv[1];
   char const* routingkey = argv[2];
 
   cJSON* root=cJSON_CreateObject();	
   cJSON_AddItemToObject(root, "name", cJSON_CreateString("Jack (\"Bee\") Nimble"));
   cJSON* fmt=cJSON_CreateObject();
   cJSON_AddItemToObject(root, "format", fmt );
   cJSON_AddStringToObject(fmt,"type","rect");
   cJSON_AddNumberToObject(fmt,"width",1920);
   cJSON_AddNumberToObject(fmt,"height",1080);
   cJSON_AddFalseToObject (fmt,"interlace");
   cJSON_AddNumberToObject(fmt,"frame rate",24);
	
   // Print to text, Delete the cJSON, print it, release the string.
   char* out=cJSON_Print(root);	
   cJSON_Delete(root);	
   printf("%s\n",out);	

   notifymq_sendstring( exchange , routingkey , out );
   free(out);	

   if((rc = notifymq_cleanup())){
      fprintf(stderr, "ABORT: notifymq_cleanup failed rc : %d \n", rc );
      return rc ;
   }
   return 0;
}
