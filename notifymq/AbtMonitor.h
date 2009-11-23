#ifndef abtmonitor_h 
#define abtmonitor_h

#include "MQ.h"
#include "TThread.h"
#include "TObjString.h"
#include "TObject.h"
#include "TString.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"


class AbtMonitor : public TNamed  {
   public:
      AbtMonitor( const char* name  , const char* title );
      virtual ~AbtMonitor();

      Bool_t      IsFinished(){ return fFinished ; }
      Bool_t      IsUpdatedEvt(){ return fEvtUpdated ; }
      Bool_t      IsUpdatedRun(){ return fRunUpdated ; }

      void  SetEvent( AbtEvent* evt );
      void  SetRun(   AbtRunInfo* run );
      AbtEvent*   GetEvent();
      AbtRunInfo* GetRun();

      void Start(); 
      static void* HandleMQ(void *arg);
      static int Demo();
      static AbtMonitor* Create();

   private:
      TThread*  fWait ;
      AbtEvent* fEvent ;
      AbtRunInfo* fRun ;

      Bool_t  fFinished ;
      Bool_t  fEvtUpdated ;
      Bool_t  fRunUpdated ;

   ClassDef( AbtMonitor , 0 )
};


R__EXTERN AbtMonitor* gMon ;

#endif 
