
#include "EvMQ.h"

#include "TSystem.h"
#include "MQ.h"
#include "TTimer.h"
#include "TObject.h"

ClassImp(EvMQ);


EvMQ::EvMQ( const char* key ) : fKey(key), fMQ(NULL), fTimer(NULL), fObj(NULL) {

    if (gSystem->Load("librootmq" ) < 0) gSystem->Exit(10);
    if (gSystem->Load("libAbtDataModel" ) < 0) gSystem->Exit(10);
 
    fTimer = new TTimer(1000) ;
    fMQ = MQ::Create() ;

    fTimer->Connect("TurnOn()"   , "EvMQ" , this , "On()");
    fTimer->Connect("Timeout()"  , "EvMQ" , this , "Check()");
    fTimer->Connect("TurnOff()"  , "EvMQ" , this , "Off()");

    fTimer->TurnOn();
}

EvMQ::~EvMQ(){
   Printf("EvMQ::~EvMQ \n");
}


void EvMQ::Check(){
    Printf("EvMQ.Check : looking for updates  \n");
    if(fMQ->IsMonitorRunning()){
        if(fMQ->IsUpdated(fKey)){
           TObject* obj = fMQ->Get( fKey, 0);
           if(obj){
               Printf("EvMQ::Check finds update in queue %s \n" ,fKey );  
               obj->Print("");
               fObj = obj ;
           }
        }
    } 
}

void EvMQ::On(){
    Printf("EvMQ::On : starting monitor thread \n");
    fMQ->StartMonitorThread();
}
 
void EvMQ::Off(){
    Printf("EvMQ::Off \n");
}
 
void EvMQ::Stop(){
    fTimer->TurnOff();
}

void EvMQ::Print(Option_t* opt="") const  {
    fMQ->Print(opt);
}







