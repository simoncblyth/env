
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include "notifymq.h"

#include <iostream>
using namespace std ;


MQ::MQ(  const char* exchange , const char* exchangetype , const char* queue , const char* routingkey, Bool_t passive , Bool_t durable , Bool_t auto_delete , Bool_t exclusive )
{

   fExchange = exchange ;
   fExchangeType = exchangetype ;
   fQueue    = queue ;
   fRoutingKey = routingkey ;

   //gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));
   
   int rc ;
   if((rc = notifymq_init())){
      fprintf(stderr, "ABORT: notifymq_init failed rc : %d \n", rc );
      exit(rc);
   }
 
   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 
} 

MQ::~MQ()
{
   notifymq_cleanup();
}

void MQ::Send( TObject* obj )
{
   TMessage *tm = new TMessage(kMESS_OBJECT);
   tm->WriteObject(obj);
   char *buffer     = tm->Buffer();
   int bufferLength = tm->Length();
   cout << "MQ::Send : serialized into buffer of length " << bufferLength << endl ;
   notifymq_sendbytes( fExchange.Data() , fRoutingKey.Data() , buffer , bufferLength );
}  

TObject* MQ::Receive( const void *msgbytes , size_t msglen )
{
   printf("MQ::Receive \n" );
   MyTMessage* mtm = new MyTMessage((void*)msgbytes,msglen);
   TClass* kls = mtm->GetClass();
   TObject* obj = mtm->ReadObject(kls);
   obj->Print();

   TString kln = obj->ClassName();
   cout << kln.Data() << endl ;
   return obj ;
}

void MQ::Wait(receiver_t handler)
{
   notifymq_basic_consume( fQueue.Data() , handler );
}


