#ifndef EVMQ_h
#define EVMQ_h

#include "Rtypes.h"
#include "TObject.h"
#include <RQ_OBJECT.h>


using namespace std ;
#include <string>
#include <map>


class EvMQ ;
class MQ ;
class TTimer ;
class CaptureDB ;

#include "TSysEvtHandler.h"
class EvTerminationHandler : public TSignalHandler {
   EvMQ*  fMon ;
public:
   EvTerminationHandler(EvMQ* mon ) : TSignalHandler(kSigTermination, kFALSE), fMon(mon) {}
   Bool_t  Notify();
};



class EvMQ : public TObject {
    /*
        Translation from evmq.py 

        Timer based architecture to handle controlled maximum frequency 
        updating of an event display 

        As new event messages can arrive faster than would want to 
        display them, establish a "pulse" via a timer allowing new messages 
        to be checked for every second (or so) while still providing 
        interactive ipython 
    */
    
    RQ_OBJECT("EvMQ")
    
    private :
        TObjArray* fKeys ; 
        MQ*     fMQ ;
        TTimer* fTimer ; 
        TObject* fObj ;
        CaptureDB* fDB ;
    
        map <string, int> fUpdates ;
        map <string, int> fChecks ;
    
    public:

        EvMQ(  const char* keys  = "default.routingkey abt.test.string abt.test.runinfo abt.test.event abt.test.other"  );
       ~EvMQ();

       void Launch();

       void On();
       void Off();

       void SetObj(TObject* obj);
       TObject* GetObj();

       void HandleTermination();
       
       void Check_( const char* key );
       void Check();
       void Verify();
       void Print(Option_t* opt="" ) const ;


    ClassDef(EvMQ , 0) // Message Queue Monitor
};

#endif




