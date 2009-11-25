
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include  "TThread.h"
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

   fMonitor = NULL ;
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
   // this segments when an instance of a class with a dictionary that is not available to be loaded is received 
   printf("MQ::Receive callback \n" );
   MyTMessage* msg = new MyTMessage((void*)msgbytes,msglen);
   TObject* obj = NULL ;

   char str[MQ::bufmax] ; 
   if (msg->What() == kMESS_STRING) {
       msg->ReadString( str , MQ::bufmax ); 
       cout << "got string " << str << endl ;
       obj = new TObjString( str );
   } else if (msg->What() == kMESS_OBJECT ){
       cout << "got object " << endl ;
       TClass* kls = msg->GetClass();
       cout << "got kls " << kls << endl ;
       kls->Print();
       cout << "read object " << endl ;   // hanging here 
       obj = msg->ReadObject(kls);
       cout << "print object " << endl ;
       obj->Print();
       cout << "get kln  " << endl ;
       TString kln = obj->ClassName();
       cout << kln.Data() << endl ;
   }
   return obj ;
}

void MQ::Wait(receiver_t handler , void* arg )
{
   if(!fConfigured) this->Configure();
   notifymq_basic_consume( fQueue.Data() , handler , arg  );
}


int MQ::handlebytes( void* arg , const void *msgbytes , size_t msglen )
{
   cout <<  "handlebytes received msglen "  << msglen << endl ; 
   TObject* obj = MQ::Receive( msgbytes , msglen );
   if ( obj == NULL ){
       cout << "received NULL obj " << endl ;
   } else {
       TString kln = obj->ClassName();
       cout << kln.Data() ; 
       if( kln == "TObjString" ){      
          cout << ((TObjString*)obj)->GetString() << endl; 
     //  } else if( kln == "AbtRunInfo" ){       
     //     ((AbtRunInfo*)obj)->Print() ;
     //  } else if ( kln == "AbtEvent" ){     
     //     ((AbtEvent*)obj)->Print() ;
       } else {
          cout << "SKIPPING received obj of class " << kln.Data() << endl ;
       }
   }
   return 0; 
}




void MQ::SetMessage(MyTMessage* msg )
{
   fMessage = msg ;
   fMessageUpdated = kTRUE ;
}

MyTMessage* MQ::GetMessage()
{
   fMessageUpdated = kFALSE ;
   return fMessage ;
}

int MQ::receive_message( void* arg , const void *msgbytes , size_t msglen )
{
   printf("MQ::Receive callback \n" );
   MyTMessage* msg = new MyTMessage((void*)msgbytes,msglen);
   MQ* instance = (MQ*)arg ;
   instance->SetMessage( msg );
   return 0 ;
}

void* MQ::Monitor(void* arg )
{
   MQ* inst = (MQ*)arg ;
   Int_t id=TThread::SelfId(); // get pthread id
   TThread::Printf("MQ::Monitor from inside thread %d ", id );
   notifymq_basic_consume( inst->fQueue.Data() , MQ::receive_message , arg  );
   return 0 ;
}

void MQ::StartMonitorThread()
{
   if(!fConfigured) this->Configure();
   fMonitor =  new TThread("monitor", (void(*) (void *))&Monitor , (void*) this);    
   fMonitor->Run();
}

void MQ::test_handlebytes()
{
    MQ* q = new MQ ;
    q->Wait( MQ::handlebytes , (void*)q );
    delete q ;
}





