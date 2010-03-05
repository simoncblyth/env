
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

ClassImp(MQ);

char* mq_frombytes(amqp_bytes_t b)
{ 
    if(b.len > 10000){
       printf("mq_frombytes probable corruped amqp_bytes ... \n");
       return NULL ;
    }

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
         << " nodestamp " 
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

void MQ::SetDebug( Int_t debug )
{
    fDebug = debug ;    
}

Int_t MQ::GetDebug()
{
    return fDebug ;
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
   Int_t dbg          =         atoi( private_lookup_default( "NOTIFYMQ_DBG" ,   "0" ) );

   gMQ->SetOptions( passive, durable, auto_delete, exclusive ) ;
   
   if( dbg > 0 ) gMQ->Print() ;
   gMQ->SetDebug( dbg );

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

   fDebug = 0 ;

   this->SetOptions();      // take the defaults initially , change using SetOptions before any actions
   fConfigured = kFALSE ;
   fMonitorRunning = kFALSE ;
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

   // observer is invoked (from inside the lock)
   //  when messages are added with fRoutingKey ... caution what you call otherwise deadlock
   //  just use observer to signal the update  
   ConfigureQueue( fRoutingKey.Data() , QueueObserver , (void*)this , 5  );  

}


MQ::~MQ()
{
   notifymq_cleanup();
}

void MQ::SendRaw( const char* str , const char* key )
{
   if(!fConfigured) this->Configure();
   const char* ukey = key == NULL ? fRoutingKey.Data() : key ; 
   notifymq_sendstring( fExchange.Data() , ukey  , str );
}

void MQ::SendString( const char* str , const char* key  )
{
   TMessage *tm = new TMessage(kMESS_STRING);
   tm->WriteCharP(str);
   this->SendMessage( tm , key );
}

void MQ::SendObject( TObject* obj , const char* key  )
{
   TMessage *tm = new TMessage(kMESS_OBJECT);
   tm->WriteObject(obj);
   this->SendMessage( tm , key );
}

void MQ::SendMessage( TMessage* msg , const char* key   )
{
   if(!fConfigured) this->Configure();
   char *buffer     = msg->Buffer();
   int bufferLength = msg->Length();

   const char* ukey = key == NULL ? fRoutingKey.Data() : key ; 
   cout << "MQ::SendMessage : serialized into buffer of length " << bufferLength << " ukey : \"" << ukey << "\"" << endl ;
   notifymq_sendbytes( fExchange.Data() , ukey  , buffer , bufferLength );
}


void MQ::SendJSON(TClass* kls, TObject* obj , const char* key )
{
   // the interface cannot be reduced as you might imagine as : ((TObject*)ri)->Class() != ri->Class()
   cJSON* o = root2cjson( kls , obj );
   char* out = cJSON_Print(o) ;
   cout << "MQ::SendJSON " << out << endl ;
   this->SendRaw( out , key );
}


int MQ::QueueObserver( void* me , const char* key ,  notifymq_collection_qstat_t* qstat )
{
   //
   //  Could have msg arg too ?   notifymq_basic_msg_t* msg 
   //  BUT cannot do much from here as this is
   //  executed from inside the monitoring thread ...
   //
   //  so avoid doing anything involved ... such as creating a TObject from a message
   //
   //  setup the signal such that it propagates the arguments needed 
   //  to construct the corresponding object via absolute addressing 
   //    ( key , index ) ... 
   //


   MQ* self = (MQ*)me ; 
   
   Int_t dbg = self->GetDebug();
   if(dbg > 0){
        cout << "MQ::DemoObserver key [" << key << "] qstat " 
        << " read:" << qstat->read 
        << " received:" << qstat->received 
        << " lastread:" << qstat->lastread 
        << " lastadd:" << qstat->lastadd 
        << " updated:" << qstat->updated 
        << " msgmax:"  << qstat->msgmax 
        <<  endl ;
        self->Print();
    }
   //self->QueueUpdated();
   self->QueueUpdatedIndex( (Long_t)qstat->lastadd );
   // it would be nice to pass the struct in the signal 
   return 42 ;
}


void MQ::QueueUpdatedIndex( Long_t index )
{
   Emit("QueueUpdatedIndex(Long_t)", index);
}
void MQ::QueueUpdated()
{
   Emit("QueueUpdated()");
}

void MQ::ConfigureQueue( const char* key , notifymq_collection_observer_t obs, void* args , int msgmax  )
{
   notifymq_collection_queue_configure( key , obs , args , msgmax  ); 
}

notifymq_collection_qstat_t MQ::QueueStat( const char* key )
{
   return notifymq_collection_queue_stat( key ); 
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
    
    const char* type     =  notifymq_get_content_type( msg );
    const char* encoding =  notifymq_get_content_encoding( msg );

    Int_t dbg = this->GetDebug();
    if(dbg > 1) cout << "MQ::Get index " << msg->index << " type " << type << " encoding " << encoding << endl ;

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


Bool_t MQ::IsMonitorRunning()
{
   return fMonitorRunning ;
}

void MQ::StartMonitorThread()
{
   if(!fConfigured) this->Configure();
   fMonitorRunning = kTRUE ;
   notifymq_basic_consume_async( fQueue.Data() );
}

void MQ::StopMonitorThread()
{
   fMonitorRunning = kFALSE ;
}



