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
   cJSON_AddNumberToObject(root, "number", 309);
   cJSON_AddStringToObject(root, "start", "2009-09-10 01:58:52");
   cJSON_AddStringToObject(root, "stop",  "2009-09-10 01:59:05");
   cJSON_AddNumberToObject(root, "events" , 0 );
   cJSON_AddNumberToObject(root, "operator" , 0 );
   cJSON_AddNumberToObject(root, "tkoffset" , 0 );
   cJSON_AddNumberToObject(root, "source  " , 0 );
   cJSON_AddNumberToObject(root, "pmtgain" , 0.0 );
   cJSON_AddNumberToObject(root, "trigger" , 0 );
   cJSON_AddNumberToObject(root, "temperature" , 25.0 );
   cJSON_AddNumberToObject(root, "humidity" , 75.0 );
   cJSON_AddStringToObject(root, "comment" ,  "Calibration Run" );
   cJSON_AddStringToObject(root, "frontendhost" ,  "dayabay8core" );
   cJSON_AddStringToObject(root, "frontendname" ,  "Frontend" );
   cJSON_AddStringToObject(root, "created" ,       "2009-11-12 19:03:21.676917" );

   // NB a dict with more structure that this rather flat one can also be prepared, using AddItemToObject :
/*
   cJSON* root=cJSON_CreateObject();	
   cJSON_AddItemToObject(root, "name", cJSON_CreateString("Jack (\"Bee\") Nimble"));
   cJSON* fmt=cJSON_CreateObject();
   cJSON_AddItemToObject(root, "format", fmt );
   cJSON_AddStringToObject(fmt,"type","rect");
   cJSON_AddNumberToObject(fmt,"width",1920);
   cJSON_AddNumberToObject(fmt,"height",1080);
   cJSON_AddFalseToObject (fmt,"interlace");
   cJSON_AddNumberToObject(fmt,"frame rate",24);
*/

	
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
