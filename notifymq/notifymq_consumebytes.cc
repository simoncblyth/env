#include <stdio.h>
#include <iostream>
using namespace std ;

#include "notifymq.h"
#include "mytmessage.h"
#include "TObject.h"
#include "TClass.h"

#include "AbtRunInfo.h"


int handlebytes( const void *msgbytes , size_t msglen )
{
   // Error in <TClass::Load>: dictionary of class AbtRunInfo not found
   //     how to dynamically load the right lib for class corresponding to the message received 
   //     ... if not might as well position these handlers in the corresponding libs 

   printf("inside handlebytes\n" );
   MyTMessage* mtm = new MyTMessage((void*)msgbytes,msglen);
   TClass* kls = mtm->GetClass();
   TObject* obj = mtm->ReadObject(kls);
   obj->Print();
   AbtRunInfo* ari = (AbtRunInfo*)obj ;
   cout << ari->GetExptName() << " " << ari->GetRunNumber() << endl ;

   return 0; 
}

int main(int argc, char const * const *argv) {

   //if (argc < 4) {
   //   fprintf(stderr, "Usage: notifymq_consume_bytes exchange routingkey messagebody\n");
   //   return 1;
   //}

   int rc ;
   if((rc = notifymq_init())){
      fprintf(stderr, "ABORT: notifymq_init failed rc : %d \n", rc );
      return rc ;
   }
   const char* exchange = "t.exchange" ;
   const char* exchangetype = "direct" ;
   const char* routingkey = "t.key" ; 
   const char* queue = "t.queue" ;

   int passive = 0 ;
   int durable = 0 ;
   int exclusive = 0 ;
   int auto_delete = 1 ;

   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 
   notifymq_basic_consume( queue , handlebytes );
 
   notifymq_cleanup();
   return 0;
}

  

