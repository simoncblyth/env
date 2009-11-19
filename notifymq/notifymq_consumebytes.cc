#include <stdio.h>
#include <iostream>
using namespace std ;

#include "notifymq.h"
#include "mytmessage.h"
#include "TObject.h"
#include "TClass.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"
#include "AbtResponse.h"


void dump_runinfo( AbtRunInfo* ari )
{
   cout << ari->GetExptName() << " " << ari->GetRunNumber() << endl ;
} 

void dump_response( AbtResponse* res )
{
   Int_t n = 16 ;
   for(Int_t i = 0 ; i < n ; i++ ){
      cout << res->GetCh(i) << " " ;
   } 
   cout << endl ;
}

void dump_event( AbtEvent* evt )
{
   cout << evt->GetSerialNumber() << endl ;
   TObject* o = NULL ; 
   TIter next(evt->GetEvtObjects()) ;
   while(( o = (TObject*)next() )){
      cout << o->GetName() << " " << o->ClassName() << endl ;
      TString kln = o->ClassName();
      if( kln == "AbtResponse" ){
          dump_response( (AbtResponse*)o ); 
      }
   }
}


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

   TString kln = obj->ClassName();
   cout << kln.Data() ; 
   if( kln == "AbtRunInfo" ){       
       dump_runinfo((AbtRunInfo*)obj) ;
   } else if ( kln == "AbtEvent" ){     
       dump_event((AbtEvent*)obj) ;
   } else {
       cout << "SKIPPING received obj of class " << kln.Data() << endl ;
   }


   // how to get the object out of this callback to somewhere it can be used ...
   // signal/slot ... delegate object ?  into pyROOT ?
   //  http://root.cern.ch/phpBB2/viewtopic.php?t=8325&highlight=tthread
   //
   // EvConsumer in separate thread ? that does the waiting around for messages
   // http://root.cern.ch/root/html/TThread.html#TThread:TThread


   return 0; 
}

int main(int argc, char const * const *argv) {

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

  

