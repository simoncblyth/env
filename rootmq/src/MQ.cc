
#include "MQ.h"

#include  "TSystem.h"
#include  "MyTMessage.h"
#include  "TMessage.h"
#include  "TClass.h"

#include "rootmq_collection.h"
#include "rootmq_utils.h"

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
   const char* type = fConsumer ? "CONSUMER" : "PRODUCER" ;    
   stringstream ss ;
   ss    << "MQ::Summary "
         << type 
         << " exchange: " << fExchange.Data() 
         << " exchangeType: " << fExchangeType.Data() 
         << " queue: " << fQueue.Data() 
         << " routingKey: " << fRoutingKey.Data() << endl 
         << " passive: " << fPassive 
         << " durable: " << fDurable 
         << " autoDelete: " << fAutoDelete 
         << " exclusive: " << fExclusive
         << " nodestamp " 
         << MQ::NodeStamp()
        ;

   string smry = ss.str();
   return mq_cstring_dupe( (void*)smry.data() , smry.length() ) ;
}


const char* MQ::GetExchange(){
    return fExchange.Data();
}
const char* MQ::GetExchangeType(){
    return fExchangeType.Data();
}
const char* MQ::GetQueue(){
    return fQueue.Data();
}
const char* MQ::GetRoutingKey(){
    return fRoutingKey.Data();
} // the default routing key if not specified otherwise



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

Bool_t MQ::IsConsumer(){ 
    return fConsumer ; 
}


MQ* MQ::Create(Bool_t consumer )
{
   /*
                         A      B      C      D
        -----------------------------------------------                 
         passive         0      0      0      0
         durable         0      1      0      1
         auto_delete     1      0      1      0
         exclusive       0      0      0      0
         
            Durable message queues continue to exist and collect messages whether or 
            not there are consumers to receive them
         
            The defaults were changed to B (with durable:1) while debugging the fanout config...
         
            Back to D when debugging topic exchange usage 
         
   */    
    
   if( gMQ != 0 ){
      cout << "MQ::Create WARNING  non null gMQ ... you can only call this once " << endl ;
      return gMQ ;
   }

   int rc = private_init();
   if(rc){
       Printf("MQ::Create error in private_init\n") ; 
       gSystem->Exit(rc);
   }

   const char* hostname     = gSystem->HostName() ;
   const char* exchange     = private_lookup_default( "ROOTMQ_EXCHANGE" ,    "abt" );
   const char* queue        = private_lookup_default( "ROOTMQ_QUEUE" ,       hostname );
   const char* routingkey   = private_lookup_default( "ROOTMQ_ROUTINGKEY" ,  "abt.#" );
   const char* exchangetype = private_lookup_default( "ROOTMQ_EXCHANGETYPE", "topic" );
   
   gMQ = new MQ( consumer, exchange, queue, routingkey, exchangetype);

   Bool_t passive     = (Bool_t)atoi( private_lookup_default( "ROOTMQ_PASSIVE" , "0" ) );
   Bool_t durable     = (Bool_t)atoi( private_lookup_default( "ROOTMQ_DURABLE" , "1" ) );   
   Bool_t auto_delete = (Bool_t)atoi( private_lookup_default( "ROOTMQ_AUTODELETE" , "0" ) ); 
   Bool_t exclusive   = (Bool_t)atoi( private_lookup_default( "ROOTMQ_EXCLUSIVE" , "0" ) );

   gMQ->SetOptions( passive, durable, auto_delete, exclusive ) ;
   
   Int_t dbg          =         atoi( private_lookup_default( "ROOTMQ_DBG" ,   "0" ) );
   if( dbg > 0 ) gMQ->Print() ;
   gMQ->SetDebug( dbg );
   
   if( gMQ->IsConsumer() ){
       gMQ->StartMonitorThread() ;
   }
   private_cleanup();
   return gMQ ;
}  


MQ::MQ(  Bool_t consumer, const char* exchange ,  const char* queue , const char* routingkey , const char* exchangetype ) 
{
   if (gMQ != 0)
      throw("There can be only one!");
   gMQ = this;

   fConsumer = consumer ;
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
    /*
         AMQP model ..
             1) publish to an exchange ... no queue needed
             2) consume from a queue which is bound to an exchange
    
         publishing is simpler as done syncronously : when you so desire 
         consumption is harder ... as you dont know when you will receive the messages 
    */
    
   if(fConfigured){
      Printf("MQ::Configure WARNING are already configured " );
      this->Print();
      return ;
   }

   // login and open connection to server
   int rc ;
   if((rc = rootmq_init())){
      fprintf(stderr, "ABORT: rootmq_init failed rc : %d \n", rc );
      exit(rc);
   }
   
   // producers/consumers need an exchange to send/bind to 
   rootmq_exchange_declare( fExchange.Data() , fExchangeType.Data() , fPassive , fDurable, fAutoDelete  ); 
  
   
   // only consumers need to bind a queue up to an exchange 
   // and prepare callbacks for message collection + notification onwards to GUIs etc..

   if(fConsumer){
       cout << "MQ::Configure setting up CONSUMER " <<  endl ;
       rootmq_queue_declare(    fQueue.Data(), fPassive , fDurable, fExclusive, fAutoDelete  ); 
       rootmq_queue_bind(       fQueue.Data(), fExchange.Data() , fRoutingKey.Data() );    
       Int_t maxlen = 10 ;
       CollectionConfigure( fRoutingKey.Data() , CollectionObserver , (void*)this , maxlen  );
   } else {
       cout << "MQ::Configure setting up PRODUCER " <<  endl ;   
   }
   fConfigured = kTRUE ;

}


MQ::~MQ()
{
    if(fMonitorRunning) StopMonitorThread();
}




// preparing to send

void MQ::SendAString( const char* str , const char* key  )
{
    TString astr = Form("%s : %s", NodeStamp(), str  ) ;  
    SendRaw( astr.Data() , key );
}

void MQ::SendString( const char* str , const char* key  )
{
    SendRaw( str , key );
}

void MQ::SendStringAsTMessage( const char* str , const char* key  )
{
   // such messages kill iChat consumers ... causing disconnection     
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

void MQ::SendJSON(TClass* kls, TObject* obj , const char* key )
{
   // the interface cannot be reduced as you might imagine as : ((TObject*)ri)->Class() != ri->Class()
   cJSON* o = root2cjson( kls , obj );
   char* out = cJSON_Print(o) ;
   cout << "MQ::SendJSON " << out << endl ;
   this->SendRaw( out , key );
}


// actual senders 

void MQ::SendRaw( const char* str , const char* key )
{
   if(!fConfigured) this->Configure();
   const char* ukey = key == NULL ? fRoutingKey.Data() : key ; 
   cout << "MQ::SendRaw : " << str  << " ukey : \"" << ukey << "\"" << " xchg : " << fExchange.Data() << endl ;
   rootmq_sendstring( fExchange.Data() , ukey  , str );
}

void MQ::SendMessage( TMessage* msg , const char* key   )
{
   if(!fConfigured) this->Configure();
   char *buffer     = msg->Buffer();
   int bufferLength = msg->Length();

   const char* ukey = key == NULL ? fRoutingKey.Data() : key ; 
   cout << "MQ::SendMessage : serialized into buffer of length " << bufferLength << " ukey : \"" << ukey << "\"" << endl ;
   rootmq_sendbytes( fExchange.Data() , ukey  , buffer , bufferLength );
}





int MQ::CollectionObserver( void* me , const char* key ,  rootmq_collection_qstat_t* qstat )
{
   /*
   
       CAUTION : 
          this is invoked from within the monitoring thread (inside the lock)
          when messages are added with fRoutingKey 
                 
            * caution what you call otherwise deadlock
            * just use observer to signal the update
            * avoid doing anything involved ... such as creating a TObject from a message
     
        setup the signal such that it propagates the arguments needed 
        to construct the corresponding object via absolute addressing 
           ( key , index ) ... 
         
   */

   MQ* self = (MQ*)me ;   // static method trick 
   Int_t dbg = self->GetDebug();
   if(dbg > 0){
        cout << "MQ::CollectionObserver key [" << key << "] qstat " 
        << " read:" << qstat->read 
        << " received:" << qstat->received 
        << " lastread:" << qstat->lastread 
        << " lastadd:" << qstat->lastadd 
        << " updated:" << qstat->updated 
        << " msgmax:"  << qstat->msgmax 
        <<  endl ;
        self->Print();
    }
   self->CollectionUpdatedIndex( (Long_t)qstat->lastadd );
   // it would be nice to pass the struct in the signal 
   return 42 ;
}


void MQ::CollectionUpdatedIndex( Long_t index )
{
   Emit("CollectionUpdatedIndex(Long_t)", index);
}
void MQ::CollectionUpdated()
{
   Emit("CollectionUpdated()");
}



void MQ::CollectionDump()
{
    cout << "MQ::CollectionDump ... for any content the monitor must have been running in order to collect messages " << endl ;
    rootmq_collection_dump();
    
    cout << "MQ::CollectionDump ... queue_lengths for each key " << endl ;
    TObjArray* keys = CollectionKeys();
    TIter next(keys);
    TObjString* s = NULL ;
    while(( s = (TObjString*)next() )){
        const char* key = s->GetString().Data();
        int len = rootmq_collection_queue_length( key );
        cout << key << " " << len << endl ;
    }
    
}

const char* MQ::CollectionKeys_( const char* re)
{
    /*
       Returns string containing all keys of messages 
       stored in the glib collection (hash of deques)
       
       The keys are delimited by single spaces.  
    
       Example usage from PyROOT :
            ROOT.gMQ.CollectionKeys_().split()
    */
    size_t bufmax = 512 ;
    char* buf = new char[bufmax];
    if( 0 == rootmq_collection_keys( buf, bufmax )) return buf ;
    return NULL ;
}

TObjArray* MQ::CollectionKeys(const char* re)
{
    const char* keys_ = CollectionKeys_(re);
    if(keys_){
        TString keys = keys_ ;
        return keys.Tokenize(" ");
    }
    return NULL ;
}

void MQ::CollectionConfigure( const char* key , rootmq_collection_observer_t obs, void* args , int msgmax  )
{
   rootmq_collection_queue_configure( key , obs , args , msgmax  ); 
}

rootmq_collection_qstat_t MQ::CollectionStat( const char* key )
{
   return rootmq_collection_queue_stat( key ); 
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
    if(!fMonitorRunning) return kFALSE ;
    Bool_t r = (Bool_t)rootmq_collection_queue_updated( key );
    //cout << "MQ::IsUpdated [" << key << "] " << r << endl ;
    return r;
}

Int_t MQ::GetAccessed( const char* key , int n )
{
    return (Int_t)rootmq_collection_accessed( key , n  );
}

Int_t MQ::GetLength( const char* key )
{
    return (Int_t)rootmq_collection_queue_length( key );
}
Int_t MQ::GetMaxLength( const char* key )
{
    return (Int_t)rootmq_collection_queue_get_maxlen( key );
}
void MQ::SetMaxLength( const char* key , int maxlen )
{
    rootmq_collection_queue_set_maxlen( key , maxlen );
}
TObject* MQ::Pop( const char* key , int n )
{
    rootmq_basic_msg_t* msg = rootmq_collection_pop( key , n );  
    return MQ::ConvertMessage( msg );   
} 

TObject* MQ::Peek( const char* key , int n ) 
{
    /*
       only difference between Get and Peek is the recording of the access
    */
    rootmq_basic_msg_t* msg = rootmq_collection_peek( key , n , 0 );  
    return MQ::ConvertMessage( msg );
}

TObject* MQ::Get( const char* key , int n ) 
{
    rootmq_basic_msg_t* msg = rootmq_collection_get( key , n );  
    return MQ::ConvertMessage( msg );
}
    
/*
   CAUTION wrt  ::Peek and ::Get
      the msg accessed is owned by the collection and
      will get popped off the tail and deallocated as the collection fills up
      ... perhaps should dupe ? but the TMessage ctor does that anyhow
      it still remains theoretically possible for this to walk into undefined territory 
*/


TObject* MQ::ConvertMessage( rootmq_basic_msg_t* msg )
{
    TObject* obj = NULL ;
    if(!msg) return obj ;
       
    Int_t dbg = this->GetDebug();
    const char* type     =  rootmq_get_content_type( msg );
    const char* encoding =  rootmq_get_content_encoding( msg );
    
    if( strcmp( type , "application/data" ) == 0 && strcmp( encoding , "binary" ) == 0 ){
       obj = MQ::Receive( msg->body.bytes , msg->body.len );
    } else if ( strcmp( type , "text/plain" ) == 0 ){
       char* str = mq_frombytes( msg->body );
       obj = new TObjString( str );
    } else {
       cout << "MQ::ConvertMessage WARNING unknown (type,encoding) : (" << type << "," << encoding << ")" << endl ;
    }
    
    if(dbg > 1) cout << "MQ::ConvertMessage key " << msg->key << " index " << msg->index << " type " << type << " encoding " << encoding << endl ;
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
   rootmq_basic_consume_async( fQueue.Data() );
   //rootmq_basic_consume( fQueue.Data() );  // ....  dont spin off thread (means that GUI doesnt update) ... BUT useful to check if threading is the cause of issues
}

void MQ::StopMonitorThread()
{
   fMonitorRunning = kFALSE ;
   if(fDebug > 0) Printf("MQ::StopMonitorThread start cleanup ... ");
   rootmq_terminate();
   if(fDebug > 0) Printf("MQ::StopMonitorThread completed cleanup ");
}



