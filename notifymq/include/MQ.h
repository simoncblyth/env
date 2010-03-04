#ifndef MQ_h
#define MQ_h


#include <RQ_OBJECT.h>
#include  "TObject.h"
#include  "TString.h"
#include  "notifymq.h"

class TMessage ;
class TClass ;
class MyTMessage ;

class MQ : public TObject {

  RQ_OBJECT("MQ")

  private:
        TString fExchange ;  
        TString fExchangeType ;  
        TString fQueue ;  
        TString fRoutingKey ;  

        Bool_t  fPassive ;
        Bool_t  fDurable ;
        Bool_t  fAutoDelete ;
        Bool_t  fExclusive  ;
        Bool_t  fConfigured ;
        
        Int_t   fDebug ;

        Bool_t  fMonitorRunning ;

  public:
     MQ( const char* exchange = "t.exchange" , 
         const char* queue = "t.queue" , 
         const char* routingkey = "t.key" ,
         const char* exchangetype = "direct" );
     virtual ~MQ();


     static MQ* Create(Bool_t start_monitor=kFALSE);
     void SetOptions( 
                      Bool_t passive = kFALSE ,     // passive: the exchange/queue will not get declared but an error will be thrown if it does not exist.
                      Bool_t durable = kFALSE ,     // durable: the exchange/queue will survive a broker restart. 
                      Bool_t auto_delete = kTRUE ,  // auto-delete: the exchange/queue will get deleted as soon as there are no more queues/subscriptions bound to it. 
                      Bool_t exclusive = kFALSE );  // exclusive: there can only be one client for this specific queue.
     // these defaults are taken initially, to use others settings call SetOptions before sending anything 
     void Configure();

     Bool_t IsUpdated( const char* key );
     Int_t GetLength( const char* key );
     TObject* Get( const char* key , int n );
     static TObject* Receive( void* msgbytes , size_t msglen );
     Bool_t IsMonitorRunning();

     void SendJSON(TClass* kls, TObject* obj , const char* key = NULL );
     void SendObject(TObject* obj , const char* key = NULL );
     void SendString(const char* str , const char* key = NULL );
     void SendRaw(const char* str , const char* key = NULL );
     void SendMessage(TMessage* msg , const char* key = NULL );

     void ConfigureQueue( const char* key , notifymq_collection_observer_t obs , void* obsargs , int msgmax );
     static int QueueObserver( void* me , const char* key, notifymq_collection_qstat_t* args );  
     void QueueUpdatedIndex( Long_t index ); /* SIGNAL */
     void QueueUpdated(); /* SIGNAL */
     notifymq_collection_qstat_t QueueStat( const char* key );

     void SetDebug(Int_t debug);
     Int_t GetDebug();
     void StartMonitorThread();
     void StopMonitorThread();       

     static const char* NodeStamp();
     char* Summary() const;
     void Print(Option_t *option = "") const;

     ClassDef(MQ , 0) // Message Queue 
};


R__EXTERN MQ* gMQ ;

#endif

