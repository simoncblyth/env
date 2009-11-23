#ifndef MQ_h
#define MQ_h

#include  "TObject.h"
#include  "TString.h"
#include  "notifymq.h"

class TMessage ;
class TClass ;

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

  public:
     MQ( const char* exchange = "t.exchange" , 
         const char* queue = "t.queue" , 
         const char* routingkey = "t.key" ,
         const char* exchangetype = "direct" );

     void SetOptions( Bool_t passive = kFALSE , 
                   Bool_t durable = kFALSE , 
                   Bool_t auto_delete = kTRUE , 
                   Bool_t exclusive = kFALSE ); 
     // these defaults are taken initially, to use others settings call SetOptions before sending anything 
     void Configure();

     virtual ~MQ();

     void SendJSON(TClass* kls, TObject* obj );
     void SendObject(TObject* obj );
     void SendString(const char* str );
     void SendRaw(const char* str );
     void SendMessage(TMessage* msg );

     void Wait(receiver_t handler);
     static TObject* Receive( const void *msgbytes , size_t msglen );
     static Int_t bufmax ;
     static int handlebytes( const void *msgbytes , size_t msglen );
     static void test_handlebytes();
 

     void Print(Option_t *option = "") const;

     static MQ* Create();


     ClassDef(MQ , 0) // Message Queue 
};


R__EXTERN MQ* gMQ ;

#endif

