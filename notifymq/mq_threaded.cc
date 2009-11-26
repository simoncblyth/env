
#include "MQ.h"
#include "notifymq.h"
#include "TThread.h"
#include "TObjString.h"
#include "TObject.h"
#include "TString.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"

static AbtEvent* evt = NULL ;
static AbtRunInfo* run = NULL ;
static Bool_t finished = kFALSE ;
static Bool_t evt_updated = kFALSE ;
static Bool_t run_updated = kFALSE ;

int handlebytes( void* arg , const void *msgbytes , size_t msglen , notifymq_props_t props )
{
   TThread::Printf( "handlebytes received msglen %d \n" , msglen ) ; 
   TObject* obj = MQ::Receive( (void*)msgbytes , msglen );
   if ( obj == NULL ){
       TThread::Printf("received NULL obj \n" ) ;
   } else {
       TString kln = obj->ClassName();
       TThread::Printf(" class %s ", kln.Data() );
       if( kln == "TObjString" ){      
         TThread::Printf( " %s\n",  ((TObjString*)obj)->GetString().Data() ); 
       } else if( kln == "AbtRunInfo" ){       
          TThread::Lock();
          run =  (AbtRunInfo*)obj ;
          run_updated = kTRUE ;
          TThread::UnLock();
       } else if ( kln == "AbtEvent" ){ 
          TThread::Lock();
          evt =  (AbtEvent*)obj ;
          evt_updated = kTRUE ;
          TThread::UnLock();
       } else {
          TThread::Printf("SKIPPING received obj of class %s \n" , kln.Data() ) ;
       }
   }
   return 0; 
}

void *handle(void *ptr)
{
   long nr = (long) ptr;
   TThread::Printf(" starting thread handle %ld " , nr );
   MQ* q = new MQ ;
   q->Wait( handlebytes , (void*)q );
   delete q ;
   return 0 ;
}

void mq_threaded()
{
   //gSystem->Load("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.so");
   //gSystem->Load("lib/libnotifymq.so");
   // see $ROOTSYS/tutorials/thread/threadsh1.C
   TThread* t = new TThread("t0", handle , (void*)0 );
   t->Run();
   TThread::Ps();

   while(!finished){
      if( evt_updated ){
         evt->Print();
         evt_updated = kFALSE ;
      } 
      if( run_updated ){
         run->Print();
         run_updated = kFALSE ;
      } 
      gSystem->Sleep(100);
      gSystem->ProcessEvents();
   }
   TThread::Ps();

   t->Join();
   delete t ;

   //gSystem->Sleep(5 * 60 * 1000);   // there is no waiting for threads to finish ... twil die immediately without this 
}


int main(int argc, char const * const *argv) 
{
   mq_threaded();
   return 0;
}


