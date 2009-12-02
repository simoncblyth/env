
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include  "TThread.h"
#include  "TVirtualMutex.h"
#include  "TClass.h"


//#include <amqp.h>
//#include <amqp_framing.h>
//#include "notifymq.h"
#include "notifymq_collection.h"

#include "root2cjson.h"
#include "private.h"

#include <iostream>
#include <sstream>
#include <string>
using namespace std ;



char* mq_frombytes(amqp_bytes_t b)
{ 
    char* str  = new char[b.len+1];
    memcpy( str , b.bytes , b.len );
    str[b.len] = 0 ;   // null termination 
    return str ;
}

char* mq_cstring_dupe( void* bytes , size_t len )
{ 
    char* str  = new char[len+1];
    memcpy( str , bytes , len );
    str[len] = 0 ;   // null termination 
    return str ;
}


MQ* gMQ = 0 ;

const char* MQ::NodeStamp()
{
    const size_t max = 256 ;
    char* stamp  = new char[max+1];
    stamp[max] = 0;
    const char* afmt = "%s@%s %s" ;
    const char* tfmt1 = "%c" ;
    private_getuserhostftime( stamp , max , tfmt1 , afmt  );
    return stamp ;
}

char* MQ::Summary() const
{
   stringstream ss ;
   ss 
         << " exchange " << fExchange.Data() 
         << " exchangeType " << fExchangeType.Data() 
         << " queue " << fQueue.Data() 
         << " routingKey " << fRoutingKey.Data() 
         << " passive " << fPassive 
         << " durable " << fDurable 
         << " autoDelete " << fAutoDelete 
         << " exclusive " << fExclusive
         << " " 
         << MQ::NodeStamp()
        ;

   string smry = ss.str();
   return mq_cstring_dupe( (void*)smry.data() , smry.length() ) ;
}

void MQ::Print(Option_t* opt ) const 
{
   Printf("%s\n", Summary() ); 
}

void MQ::SetOptions(  Bool_t passive , Bool_t durable , Bool_t auto_delete , Bool_t exclusive )
{
   fPassive = passive ;
   fDurable = durable ;
   fAutoDelete = auto_delete ;
   fExclusive = exclusive ;
}

MQ* MQ::Create(Bool_t start_monitor)
{
   if( gMQ != 0 ){
      cout << "MQ::Create WARNING  non null gMQ ... you can only call this once " << endl ;
      return gMQ ;
   }

   private_init();

   const char* exchange     = private_lookup_default( "NOTIFYMQ_EXCHANGE" ,    "default.exchange" );
   const char* queue        = private_lookup_default( "NOTIFYMQ_QUEUE" ,       "default.queue" );
   const char* routingkey   = private_lookup_default( "NOTIFYMQ_ROUTINGKEY" ,  "default.routingkey" );
   const char* exchangetype = private_lookup_default( "NOTIFYMQ_EXCHANGETYPE", "direct" );

   gMQ = new MQ(exchange, queue, routingkey, exchangetype);

   Bool_t passive     = (Bool_t)atoi( private_lookup_default( "NOTIFYMQ_PASSIVE" , "0" ) );
   Bool_t durable     = (Bool_t)atoi( private_lookup_default( "NOTIFYMQ_DURABLE" , "0" ) );
   Bool_t auto_delete = (Bool_t)atoi( private_lookup_default( "NOTIFYMQ_AUTODELETE" , "1" ) );
   Bool_t exclusive   = (Bool_t)atoi( private_lookup_default( "NOTIFYMQ_EXCLUSIVE" , "0" ) );

   gMQ->SetOptions( passive, durable, auto_delete, exclusive ) ;
   gMQ->Print() ;

   if( start_monitor ){
      gMQ->StartMonitorThread() ;
   }

   private_cleanup();
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

   this->SetOptions();      // take the defaults initially , change using SetOptions before any actions
   fConfigured = kFALSE ;


   fMonitor = NULL ;
   fMQMutex = new TMutex();

   fMonitorFinished = kFALSE ;
   fBytes = NULL ;
   fLength = 0 ;
   fBytesUpdated = kFALSE ; 
   fContentType = NULL ;
   fContentEncoding  = NULL ;
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
}


MQ::~MQ()
{
   notifymq_cleanup();
   delete fMQMutex ;
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


void MQ::Wait(receiver_t handler , void* arg )
{
   if(!fConfigured) this->Configure();
   notifymq_basic_consume( fQueue.Data() , handler , arg  );
}

// private setters, protected en-masse

void MQ::SetBytesLength(void* bytes, size_t len )
{
   fBytes = bytes ;
   fLength = len ;
   fBytesUpdated = kTRUE ;
}
void MQ::SetContentType(char* str)
{
   fContentType = str  ;
}
void MQ::SetContentEncoding(char* str)
{
   fContentEncoding = str  ;
}

// private getters, as not needed to be exposed ... and thread sensitive

void* MQ::GetBytes()
{
   R__LOCKGUARD(fMQMutex);
   fBytesUpdated = kFALSE ;
   return fBytes ;
}
size_t MQ::GetLength()
{
   R__LOCKGUARD(fMQMutex);
   return fLength ;
}
char* MQ::GetContentType()
{
   R__LOCKGUARD(fMQMutex);
   return fContentType ;
}
char* MQ::GetContentEncoding()
{
   R__LOCKGUARD(fMQMutex);
   return fContentEncoding ;
}








// public getters, individually protected 

Bool_t MQ::IsMonitorFinished(){ 
   R__LOCKGUARD(fMQMutex);
   return fMonitorFinished ; 
}
Bool_t MQ::IsBytesUpdated(){    
   R__LOCKGUARD(fMQMutex);
   return fBytesUpdated  ; 
}

// private internals 


TObject* MQ::Receive( void* msgbytes , size_t msglen )
{
   MyTMessage* msg = new MyTMessage( msgbytes , msglen );
   TObject* obj = NULL ;

   if (msg->What() == kMESS_STRING) {
       char* buf = new char[msglen];  
       msg->ReadString( buf , msglen ); 
       //cout << "got string [" << buf << "]" <<  endl ;
       obj = new TObjString( buf );
   } else if (msg->What() == kMESS_OBJECT ){
       TClass* kls = msg->GetClass();
       obj = msg->ReadObject(kls);
   }
   return obj ;
}

TObject* MQ::ConstructObject()
{
   //cout << "MQ::ConstructObject (type,encoding) : (" << fContentType << "," << fContentEncoding << ")" << endl ;
   R__LOCKGUARD(fMQMutex);
   TObject* obj = NULL ;
   TString type = fContentType ;
   TString encoding = fContentEncoding ;
   if(type == "application/data" && encoding == "binary" ){
       obj = MQ::Receive( fBytes , fLength );
       fBytesUpdated = kFALSE ; 
   } else if (type == "text/plain" && encoding == "" ){
       char* str = mq_cstring_dupe( fBytes , fLength );
       fBytesUpdated = kFALSE ; 
       obj = new TObjString( str );
   } else {
       cout << "MQ::ConstructObject WARNING unknown (type,encoding) : (" << type.Data() << "," << encoding.Data() << ")" << endl ;
   }
   return obj ;
}


TObject* MQ::Get( const char* key , int n ) 
{
    TObject* obj = NULL ;
    notifymq_basic_msg_t* msg = notifymq_collection_get( key , n );
    if(!msg) return obj ;

    const char* type     =  mq_frombytes( msg->properties.content_type );
    const char* encoding =  mq_frombytes( msg->properties.content_encoding );
    cout << "MQ::Get index " << msg->index << " type " << type << " encoding " << encoding << endl ;

    if( strcmp( type , "application/data" ) == 0 && strcmp( encoding , "binary" ) == 0 ){
       obj = MQ::Receive( msg->body.bytes , msg->body.len );
    } else if ( strcmp( type , "text/plain" ) == 0 ){
       char* str = mq_frombytes( msg->body );
       obj = new TObjString( str );
    } else {
       cout << "MQ::Get WARNING unknown (type,encoding) : (" << type << "," << encoding << ")" << endl ;
    }
    return obj ;  
}


int MQ::receive_bytes( void* arg , const void *msgbytes , size_t msglen , notifymq_props_t props )
{
   //printf("MQ::receive_bytes of length %d \n" , msglen );
   MQ* instance = (MQ*)arg ;
   R__LOCKGUARD(instance->fMQMutex);
   instance->SetBytesLength( (void*)msgbytes , msglen );
   instance->SetContentType(      mq_cstring_dupe( props.content_type.bytes ,     props.content_type.len ));
   instance->SetContentEncoding(  mq_cstring_dupe( props.content_encoding.bytes , props.content_encoding.len  ));
   return 0 ;
}

void* MQ::Monitor(void* arg )
{
   MQ* inst = (MQ*)arg ;
   Long_t tid=TThread::SelfId(); // get pthread id
   TThread::Printf("MQ::Monitor from inside thread %ld ", tid );
   notifymq_basic_consume( inst->fQueue.Data() , MQ::receive_bytes , arg  );
   return 0 ;
}

void MQ::StartMonitorThread()
{
   if(!fConfigured) this->Configure();
   fMonitor =  new TThread("monitor", (void(*) (void *))&Monitor , (void*) this);    
   fMonitor->Run();
}

void MQ::StopMonitorThread()
{
   // more cleanup ? cancel points ?
   fMonitor->Kill() ;
}



