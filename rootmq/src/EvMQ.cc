
#include "EvMQ.h"

#include "TSystem.h"
#include "MQ.h"
#include "TTimer.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"

#include "Capture.h"
#include "CaptureDB.h"

#include <string>

// ouch ... dont like this ... how can i partition to avoid abtmodel dependency here ?
#include "AbtEvent.h"


Bool_t EvTerminationHandler::Notify()
{
   // Handle this interrupt
   Printf("Received SIGTERM: terminating");
   fMon->HandleTermination();
   return kTRUE;
}


ClassImp(EvMQ);


EvMQ::EvMQ( const char* keys ) : fKeys(NULL), fMQ(NULL), fTimer(NULL), fObj(NULL), fDB(NULL) {

    if (gSystem->Load("librootmq" ) < 0) gSystem->Exit(10);
    if (gSystem->Load("libAbtDataModel" ) < 0) gSystem->Exit(10);
 
    TString ks = keys ;
    fKeys = ks.Tokenize(" "); 
    fTimer = new TTimer(1000) ;
    fMQ = MQ::Create() ;

    /*
    const char* sample = "$ABERDEEN_HOME/DataModel/sample/run00027.root" ;
    const char* dbpath = gSystem->ExpandPathName(Form("%s.db",sample));
    fDB = new CaptureDB(dbpath);
    */
      
        
    fTimer->Connect("TurnOn()"   , "EvMQ" , this , "On()");
    fTimer->Connect("Timeout()"  , "EvMQ" , this , "Check()");
    fTimer->Connect("TurnOff()"  , "EvMQ" , this , "Off()");

    gSystem->AddSignalHandler( new EvTerminationHandler(this) );
     
}

void EvMQ::Launch()
{
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


void EvMQ::SetObj(TObject* obj)
{
    fObj = obj ;
}

TObject* EvMQ::GetObj()
{
    return fObj;
}


void EvMQ::Verify(){
    if(fDB==NULL) return ;
    if(fObj==NULL) return;
    if(strcmp(fObj->ClassName(),"AbtEvent")) return ;
        
    string got ;
    {
        Capture c;
        fObj->Print();
        got = c.Gotcha();
    }
     
    Int_t dbg = fMQ->GetDebug();    
    AbtEvent* evt = (AbtEvent*)fObj ;    
    const char* expect = fDB->Get("AbtEvent", evt->GetSerialNumber() );
    const char* now    = got.c_str();
    if(strcmp(expect,now)==0){
        if(dbg>1) cout << "EvMQ::Verify matches expectation " << endl ;
    } else {
        cout << "EvMQ::Verify mismatch " << endl ;
        cout << "expected:" << endl << expect << endl ;
        cout << "found:" << endl << now << endl ;
        HandleTermination();
    }
}

void EvMQ::Check_( const char* key )
{    
    
    fChecks[key] += 1;
    Int_t dbg = fMQ->GetDebug();
    if(fMQ->IsUpdated(key)){
        fUpdates[key] += 1;
        TObject* obj = fMQ->Get( key, 0);
        if(obj){
            if(dbg>0) Printf("EvMQ::Check_ : finds update in collection deque %s \n" ,key );  
            if(dbg>1) obj->Print("");
            fObj = obj ;
            Verify();
        } else {
            Printf("EvMQ::Check_ : null obj : %s", key );
        }
    } else {
        Printf("EvMQ::Check_ : not updated %s" , key );
    }
}




void EvMQ::Check(){
    Int_t dbg = fMQ->GetDebug();
    if(fMQ->IsMonitorRunning()){
        TIter next(fKeys);
        TObjString* s = NULL ;
        while(( s = (TObjString*)next() )){
            const char* key = s->GetString().Data();
            Check_(key);
        }
    } else {
        Printf("EvMQ::Check :  monitor not running");
    }
}

void EvMQ::Print(Option_t* opt="") const  {
    fMQ->Print(opt);
    
    for( map<string,int>::const_iterator i=fChecks.begin(); i!=fChecks.end(); ++i)
    {
    	  cout << (*i).first << ": " << (*i).second << endl;
    }
    
    for( map<string,int>::const_iterator i=fUpdates.begin(); i!=fUpdates.end(); ++i)
    {
    	  cout << (*i).first << ": " << (*i).second << endl;
    }
}







