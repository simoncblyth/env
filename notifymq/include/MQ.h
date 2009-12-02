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

        TVirtualMutex* fMQMutex ; 
        Bool_t  fBytesUpdated ;
        void*  fBytes ;      
        size_t fLength ;       
        char* fContentType ;
        char* fContentEncoding ;


        // private as access thread shared data ... demanding careful usage 
        void SetContentType(    char* str);      
        void SetContentEncoding(char* str);      
        void SetBytesLength(void* bytes, size_t len );      

        void* GetBytes();
        size_t GetLength();
        static int receive_bytes( void* arg , const void *msgbytes , size_t msglen , notifymq_props_t props );  // callback invoked by monitor thread 
        static void* Monitor(void* );    // runs as separate thread waiting for new messages 
 
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


     // these getters cross threads 
     TObject* Get( const char* key , int n );
     Bool_t IsMonitorFinished();
     Bool_t IsBytesUpdated();
     char* GetContentType();
     char* GetContentEncoding();
     TObject* ConstructObject();

     static TObject* Receive( void* msgbytes , size_t msglen );


     void SendJSON(TClass* kls, TObject* obj );
     void SendObject(TObject* obj );
     void SendString(const char* str );
     void SendRaw(const char* str );
     void SendMessage(TMessage* msg );


     void Wait(receiver_t handler, void* arg );

     void StartMonitorThread();
     void StopMonitorThread();       


     static const char* NodeStamp();
     char* Summary() const;
     void Print(Option_t *option = "") const;

     ClassDef(MQ , 0) // Message Queue 
};


R__EXTERN MQ* gMQ ;

#endif

