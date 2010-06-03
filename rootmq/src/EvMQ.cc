
#include "EvMQ.h"

#include "TSystem.h"
#include "MQ.h"
#include "TTimer.h"
#include "TObject.h"


Bool_t EvTerminationHandler::Notify()
{
   // Handle this interrupt
   Printf("Received SIGTERM: terminating");
   fMon->HandleTermination();
   return kTRUE;
}


ClassImp(EvMQ);


EvMQ::EvMQ( const char* key ) : fKey(key), fMQ(NULL), fTimer(NULL), fObj(NULL) {

    if (gSystem->Load("librootmq" ) < 0) gSystem->Exit(10);
    if (gSystem->Load("libAbtDataModel" ) < 0) gSystem->Exit(10);
 
    fTimer = new TTimer(1000) ;
    fMQ = MQ::Create() ;

    fTimer->Connect("TurnOn()"   , "EvMQ" , this , "On()");
    fTimer->Connect("Timeout()"  , "EvMQ" , this , "Check()");
    fTimer->Connect("TurnOff()"  , "EvMQ" , this , "Off()");
    
    gSystem->AddSignalHandler( new EvTerminationHandler(this) );
    fTimer->TurnOn();  // calls On via signal
}

EvMQ::~EvMQ(){
   Printf("EvMQ::~EvMQ \n");
}

void EvMQ::HandleTermination(){
    Printf("EvMQ::HandleTermination");
    fTimer->TurnOff(); // calls Off via signal
    Printf("EvMQ::HandleTermination ... after Off ... exiting ");
    gSystem->Exit(1);
}

void EvMQ::On(){
    Printf("EvMQ::On ... starting monitor thread ");
    fMQ->StartMonitorThread();
}
 
void EvMQ::Off(){
    Printf("EvMQ::Off .. stop monitor thread ");
    fMQ->StopMonitorThread();
}

void EvMQ::Check(){
    //Printf("EvMQ.Check : looking for updates %s ", fKey );
    if(fMQ->IsMonitorRunning()){
        if(fMQ->IsUpdated(fKey)){
           TObject* obj = fMQ->Get( fKey, 0);
           if(obj){
               Printf("EvMQ::Check : finds update in queue %s \n" ,fKey );  
               //obj->Print("");
               fObj = obj ;
           } else {
               Printf("EvMQ::Check : null obj ");
           }
        } else {
            Printf("EvMQ::Check : not updated %s" , fKey );
        }
    } else {
        Printf("EvMQ::Check :  monitor not running");
    }
}

void EvMQ::Print(Option_t* opt="") const  {
    fMQ->Print(opt);
}







