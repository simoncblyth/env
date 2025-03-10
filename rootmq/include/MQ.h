#ifndef MQ_h
#define MQ_h


#include <RQ_OBJECT.h>
#include  "TObject.h"
#include  "TString.h"
#include  "rootmq.h"

class TMessage ;
class TClass ;
class MyTMessage ;

class MQ : public TObject {

  /*
      Internally the local collection is comprised of a hash of dequeues  
          * http://library.gnome.org/devel/glib/unstable/glib-Double-ended-Queues.html
          * this collection is updated by the monitoring thread 
          * the hash is keyed on the routing keys of the messages  
          * accesses to this collection with rootmq_collection_get/peek/pop are thread locked 

      When monitoring AbtEvents 
        * want to see the latest, and dont care about missing 
           * (indeed missing events is a requirement when update frequency is too much for sensible GUI updating  )
        * need stack behaviour (ie LIFO)
            * implement with dequeues by restricting to : push_head/pop_head     
            * restrict size by pop_tail prior to push_head once reach maximum size 

      When presenting text messages 
        * want to see all messages in the order received 
        * need queue behaviour (ie FIFO)  
            * implement with dequeues by restricting to : push_head/pop_tail  
            * restrict size by pop_tail prior to push_head : this means the oldest messages get lost ... making way for the new 
            * avoid losses by configuring 

   */

  RQ_OBJECT("MQ")

  private:
        Bool_t  fConsumer ;
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
     MQ( Bool_t consumer = kFALSE , 
         const char* exchange = "t.exchange" , 
         const char* queue = "t.queue" , 
         const char* routingkey = "t.key" ,
         const char* exchangetype = "direct" );
     virtual ~MQ();


     static MQ* Create(Bool_t consumer=kFALSE);     // a consumer can produce too ... but not vice versa
     void SetOptions( 
                      Bool_t passive = kFALSE ,     // passive: the exchange/queue will not get declared but an error will be thrown if it does not exist.
                      Bool_t durable = kFALSE ,     // durable: the exchange/queue will survive a broker restart. 
                      Bool_t auto_delete = kTRUE ,  // auto-delete: the exchange/queue will get deleted as soon as there are no more queues/subscriptions bound to it. 
                      Bool_t exclusive = kFALSE );  // exclusive: there can only be one client for this specific queue.
     // these defaults are taken initially, to use others settings call SetOptions before sending anything 
     void Configure();

     Bool_t IsUpdated( const char* key );
     Int_t GetLength( const char* key );
     
     Bool_t IsConsumer();
     const char* GetExchange();
     const char* GetExchangeType();
     const char* GetQueue();
     const char* GetRoutingKey(); // the default routing key if not specified otherwise
     
     
     Int_t GetMaxLength( const char* key);
     void SetMaxLength( const char* key , int maxlen );
     Int_t GetAccessed( const char* key, int n=0 );
     
     TObject* Pop(  const char* key , int n=0 );
     TObject* Peek( const char* key , int n=0 );
     TObject* Get(  const char* key , int n=0 );
     TObject* ConvertMessage( rootmq_basic_msg_t* msg );
     static TObject* Receive( void* msgbytes , size_t msglen );
     Bool_t IsMonitorRunning();

     void SendAString(const char* str , const char* key = NULL ); // sends string prefixed with NodeStamp 

     void SendJSON(TClass* kls, TObject* obj , const char* key = NULL );
     void SendObject(TObject* obj , const char* key = NULL );
     void SendString(const char* str , const char* key = NULL );
     void SendStringAsTMessage(const char* str , const char* key = NULL );
     void SendRaw(const char* str , const char* key = NULL );
     void SendMessage(TMessage* msg , const char* key = NULL );

 
     void CollectionConfigure( const char* key , rootmq_collection_observer_t obs , void* obsargs , int msgmax );
     static int CollectionObserver( void* me , const char* key, rootmq_collection_qstat_t* args );  
     void CollectionUpdatedIndex( Long_t index ); /* SIGNAL */
     void CollectionUpdated(); /* SIGNAL */
     rootmq_collection_qstat_t CollectionStat( const char* key );
     void CollectionDump();
     TObjArray* CollectionKeys(const char* re=NULL);
     const char* CollectionKeys_(const char* re=NULL);

     // NB the Queue referred to here is not the remote MQ but the local glib collection 
     // better to rename Collection...

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

