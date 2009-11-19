
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include  "TClass.h"
#include "notifymq.h"
#include "root2cjson.h"

#include <iostream>
using namespace std ;

Int_t MQ::bufmax = 512 ; 
MQ* gMQ = 0 ;

void MQ::Print(Option_t* opt ) const 
{
    cout 
         << " exchange " << fExchange.Data() 
         << " exchangeType " << fExchangeType.Data() 
         << " queue " << fQueue.Data() 
         << " routingKey " << fRoutingKey.Data() 
         << " passive " << fPassive 
         << " durable " << fDurable 
         << " autoDelete " << fAutoDelete 
         << " exclusive " << fExclusive
         << endl ;

}

void MQ::SetOptions(  Bool_t passive , Bool_t durable , Bool_t auto_delete , Bool_t exclusive )
{
   fPassive = passive ;
   fDurable = durable ;
   fAutoDelete = auto_delete ;
   fExclusive = exclusive ;
}

MQ* MQ::Create()
{
   if (gMQ == 0) gMQ = new MQ();
   return gMQ ;
}  


MQ::MQ(  const char* exchange ,  const char* queue , const char* routingkey , const char* exchangetype ) 
{
   if (gMQ != 0)
      throw("There can be only one!");
   gMQ = this;

   fExchange = exchange ;
   fQueue    = queue ;
   fRoutingKey = routingkey ;
   fExchangeType = exchangetype ;

   this->SetOptions();      // take the defaults initially , change using Options before any actions
   fConfigured = kFALSE ;
} 
 

void MQ::Configure()
{
   if(fConfigured){
      Printf("MQ::Configure WARNING are already configured " );
      this->Print();
      return ;
   }

   int rc ;
   if((rc = notifymq_init())){
      fprintf(stderr, "ABORT: notifymq_init failed rc : %d \n", rc );
      exit(rc);
   }
   notifymq_exchange_declare( fExchange.Data() , fExchangeType.Data() , fPassive , fDurable, fAutoDelete  ); 
   notifymq_queue_declare(    fQueue.Data(), fPassive , fDurable, fExclusive, fAutoDelete  ); 
   notifymq_queue_bind(       fQueue.Data(), fExchange.Data() , fRoutingKey.Data() ); 

   fConfigured = kTRUE ;
   this->Print();
}


MQ::~MQ()
{
   notifymq_cleanup();
}

void MQ::SendRaw( const char* str )
{
   if(!fConfigured) this->Configure();
   notifymq_sendstring( fExchange.Data() ,fRoutingKey.Data() , str );
}

void MQ::SendString( const char* str )
{
   TMessage *tm = new TMessage(kMESS_STRING);
   tm->WriteCharP(str);
   this->SendMessage( tm );
}

void MQ::SendObject( TObject* obj )
{
   TMessage *tm = new TMessage(kMESS_OBJECT);
   tm->WriteObject(obj);
   this->SendMessage( tm );
}

void MQ::SendMessage( TMessage* msg   )
{
   if(!fConfigured) this->Configure();
   char *buffer     = msg->Buffer();
   int bufferLength = msg->Length();
   cout << "MQ::SendMessage : serialized into buffer of length " << bufferLength << endl ;
   notifymq_sendbytes( fExchange.Data() , fRoutingKey.Data() , buffer , bufferLength );
}


void MQ::SendJSON(TClass* kls, TObject* obj )
{
   // the interface cannot be reduced as you might imagine as : ((TObject*)ri)->Class() != ri->Class()
   cJSON* o = root2cjson( kls , obj );
   char* out = cJSON_Print(o) ;
   cout << "MQ::SendJSON " << out << endl ;
   this->SendRaw( out );
}


TObject* MQ::Receive( const void *msgbytes , size_t msglen )
{
   printf("MQ::Receive callback \n" );
   MyTMessage* msg = new MyTMessage((void*)msgbytes,msglen);
   TObject* obj = NULL ;

   char str[MQ::bufmax] ; 
   if (msg->What() == kMESS_STRING) {
       msg->ReadString( str , MQ::bufmax ); 
       cout << "got string " << str << endl ;
       obj = new TObjString( str );
   } else if (msg->What() == kMESS_OBJECT ){
       TClass* kls = msg->GetClass();
       kls->Print();
       obj = msg->ReadObject(kls);
       obj->Print();
       TString kln = obj->ClassName();
       cout << kln.Data() << endl ;
   }
   return obj ;
}

void MQ::Wait(receiver_t handler)
{
   if(!fConfigured) this->Configure();
   notifymq_basic_consume( fQueue.Data() , handler );
}


