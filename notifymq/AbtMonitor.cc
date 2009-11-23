
#include "AbtMonitor.h"

ClassImp(AbtMonitor);

AbtMonitor* gMon = NULL ;


AbtMonitor* AbtMonitor::Create()
{
   if (gMon == 0) gMon = new AbtMonitor(); 
   return gMon ;
}


AbtMonitor::AbtMonitor( const char* name , const char* title ) : TNamed( name, title ) , fWait(NULL) {}
AbtMonitor::~AbtMonitor(){
   if(!fWait) return ;
   fWait->Join();
   delete fWait ;
}


void AbtMonitor::Start(){
   // see $ROOTSYS/tutorials/thread/threadsh1.C
   fWait = new TThread("t0", HandleMQ , (void*)0 );
}


AbtEvent* AbtMonitor::GetEvent()
{
    fEventUpdated = kFALSE ;
    return fEvent ;  
}
AbtRunInfo* AbtMonitor::GetRun()
{
    fRunUpdated = kFALSE ;
    return fRun ;  
}
void AbtMonitor::SetEvent(AbtEvent* evt )
{
    fEvent = evt ;
    fEventUpdated = kTRUE ;
}
void AbtMonitor::SetRun(AbtRunInfo* run )
{
   fRun = run ;
   fRunUpdated = kTRUE ;
}


int abtmonitor_handlebytes( const void *msgbytes , size_t msglen )
{
   TThread::Printf( "abtmonitor_handlebytes received msglen %d \n" , msglen ) ; 
   TObject* obj = MQ::Receive( msgbytes , msglen );
   if ( obj == NULL ){
       TThread::Printf("received NULL obj \n" ) ;
   } else {
       TString kln = obj->ClassName();
       TThread::Printf(" class %s ", kln.Data() );
       if( kln == "TObjString" ){      
         TThread::Printf( " %s\n",  ((TObjString*)obj)->GetString().Data() ); 
       } else if( kln == "AbtRunInfo" ){       
          TThread::Lock();
          gMon->SetRun( (AbtRunInfo*)obj ) ;
          TThread::UnLock();
       } else if ( kln == "AbtEvent" ){ 
          TThread::Lock();
          gMon->SetEvent( (AbtEvent*)obj ) ;
          TThread::UnLock();
       } else {
          TThread::Printf("SKIPPING received obj of class %s \n" , kln.Data() ) ;
       }
   }
   return 0; 
}

void * AbtMonitor::HandleMQ(void *ptr)
{
   long nr = (long) ptr;
   TThread::Printf("AbtMontior::HandleMQ thread starting to wait on the MQ  %ld " , nr );
   MQ* q = new MQ ;
   q->Wait( abtmonitor_handlebytes );   // cleaner to use static method callback ... 
   delete q ;
   return 0 ;
}



int AbtMonitor::Demo()
{
   gMon->Start();

   while(!gMon->IsFinished()){
      if( gMon->IsUpdatedEvt() ){
         gMon->GetEvent()->Print();
      } 
      if( gMon->IsUpdatedRun() ){
         gMon->GetRun()->Print();
      } 
      gSystem->Sleep(100);
      gSystem->ProcessEvents();
   }
   TThread::Ps();
   return 0;
}


