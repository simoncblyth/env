#ifndef MQ_h
#define MQ_h

#include  "TObject.h"
#include  "TString.h"
#include  "notifymq.h"

class MQ : public TObject {

  private:
        TString fExchange ;  
        TString fExchangeType ;  
        TString fQueue ;  
        TString fRoutingKey ;  

  public:
     MQ( const char* exchange = "t.exchange" , 
         const char* exchangetype = "direct" , 
         const char* queue = "t.queue" , 
         const char* routingkey = "t.key" , 
         Bool_t passive = kFALSE , 
         Bool_t durable = kFALSE , 
         Bool_t auto_delete = kTRUE , 
         Bool_t exclusive = kFALSE ); 

     virtual ~MQ();
     void Send(TObject* obj );
     void Wait(receiver_t handler);
     static TObject* Receive( const void *msgbytes , size_t msglen );

     ClassDef(MQ , 0) // Message Queue 
};

#endif

