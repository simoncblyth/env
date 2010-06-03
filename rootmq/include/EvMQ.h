#ifndef EVMQ_h
#define EVMQ_h

#include "Rtypes.h"
#include "TObject.h"
#include <RQ_OBJECT.h>

class MQ ;
class TTimer ;

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
        const char* fKey ; 
        MQ*     fMQ ;
        TTimer* fTimer ; 
        TObject* fObj ;
    
    public:

        EvMQ(  const char* key  = "default.routingkey");
       ~EvMQ();

       void On();
       void Off();
       void Stop();
       
       void Check();
       void Print(Option_t* opt ) const ;


    ClassDef(EvMQ , 0) // Message Queue Monitor
};

#endif




