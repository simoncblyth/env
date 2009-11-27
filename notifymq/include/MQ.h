#ifndef MQ_h
#define MQ_h

#include  "TObject.h"
#include  "TString.h"
#include  "notifymq.h"

class TMessage ;
class TClass ;
class MyTMessage ;
class TThread ;

class MQ : public TObject {

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

        TThread* fMonitor  ;
        Bool_t  fMonitorFinished ;

        // written by monitor thread , copied into MyTMessage in main thread 
        //  ... currently have observed no deadlocks, and have no mutexes
        Bool_t  fBytesUpdated ;
        void*  fBytes ;      
        size_t fLength ;       
        char* fContentType ;
        char* fContentEncoding ;

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

     Bool_t IsMonitorFinished(){ return fMonitorFinished ; }
     Bool_t IsBytesUpdated(){  return fBytesUpdated  ; }
     void SetBytesLength(void* bytes, size_t len );      
     void* GetBytes();
     size_t GetLength();
     void SetContentType(    char* str);      
     void SetContentEncoding(char* str);      
     char* GetContentType();
     char* GetContentEncoding();
     TObject* ConstructObject();


     void SendJSON(TClass* kls, TObject* obj );
     void SendObject(TObject* obj );
     void SendString(const char* str );
     void SendRaw(const char* str );
     void SendMessage(TMessage* msg );
     static TObject* Receive( void* msgbytes , size_t msglen );
     void Wait(receiver_t handler, void* arg );

     static int receive_bytes( void* arg , const void *msgbytes , size_t msglen , notifymq_props_t props );  // callback invoked by monitor thread 
     void StartMonitorThread();
     void StopMonitorThread();       
     static void* Monitor(void* );    // runs as separate thread waiting for new messages 
     static const char* NodeStamp();
     char* Summary() const;
     void Print(Option_t *option = "") const;

     ClassDef(MQ , 0) // Message Queue 
};


R__EXTERN MQ* gMQ ;

#endif

