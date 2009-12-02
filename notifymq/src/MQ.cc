
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include  "TClass.h"

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




// private internals 


TObject* MQ::Receive( void* msgbytes , size_t msglen )
{
   MyTMessage* msg = new MyTMessage( msgbytes , msglen );
   TObject* obj = NULL ;

   if (msg->What() == kMESS_STRING) {
       char* buf = new char[msglen];  
       msg->ReadString( buf , msglen ); 
       obj = new TObjString( buf );
   } else if (msg->What() == kMESS_OBJECT ){
       TClass* kls = msg->GetClass();
       obj = msg->ReadObject(kls);
   }
   return obj ;
}



Bool_t MQ::IsUpdated( const char* key )
{
    return (Bool_t)notifymq_collection_queue_updated( key );
}
Int_t MQ::GetLength( const char* key )
{
    return (Int_t)notifymq_collection_queue_length( key );
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



void MQ::StartMonitorThread()
{
   if(!fConfigured) this->Configure();
   notifymq_basic_consume_async( fQueue.Data() );
}

void MQ::StopMonitorThread()
{
}



