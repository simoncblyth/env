#include <stdio.h>
#include <iostream>
using namespace std ;

#include "MQ.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"
#include "AbtResponse.h"


void dump_runinfo( AbtRunInfo* ari )
{
   cout << ari->GetExptName() << " " << ari->GetRunNumber() << endl ;
} 

void dump_response( AbtResponse* res )
{
   Int_t n = 16 ;
   for(Int_t i = 0 ; i < n ; i++ ){
      cout << res->GetCh(i) << " " ;
   } 
   cout << endl ;
}

void dump_event( AbtEvent* evt )
{
   cout << evt->GetSerialNumber() << endl ;
   TObject* o = NULL ; 
   TIter next(evt->GetEvtObjects()) ;
   while(( o = (TObject*)next() )){
      cout << o->GetName() << " " << o->ClassName() << endl ;
      TString kln = o->ClassName();
      if( kln == "AbtResponse" ){
          dump_response( (AbtResponse*)o ); 
      }
   }
}

int handlebytes( const void *msgbytes , size_t msglen )
{
   cout <<  "handlebytes received msglen "  << msglen << endl ; 
   TObject* obj = MQ::Receive( msgbytes , msglen );
   if ( obj == NULL ){
       cout << "received NULL obj " << endl ;
   } else {
       TString kln = obj->ClassName();
       cout << kln.Data() ; 
       if( kln == "TObjString" ){      
          cout << ((TObjString*)obj)->GetString() << endl; 
       } else if( kln == "AbtRunInfo" ){       
          dump_runinfo((AbtRunInfo*)obj) ;
       } else if ( kln == "AbtEvent" ){     
          dump_event((AbtEvent*)obj) ;
       } else {
          cout << "SKIPPING received obj of class " << kln.Data() << endl ;
       }
   }
   return 0; 
}


int main(int argc, char const * const *argv) 
{
   MQ* q = new MQ();
   q->Wait( handlebytes );
   delete q ;
   return 0;
}

  

