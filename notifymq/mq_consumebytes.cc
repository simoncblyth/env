#include <iostream>
using namespace std ;

#include "MQ.h"
#include "notifymq.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"


// callbacks need to be defined and set in compiled code, not from cint 
int handlebytes( void* arg , const void *msgbytes , size_t msglen, notifymq_props_t props )
{
   cout <<  "handlebytes received msglen "  << msglen << endl ; 
   TObject* obj = MQ::Receive( (void*)msgbytes , msglen );
   if ( obj == NULL ){
       cout << "received NULL obj " << endl ;
   } else {
       TString kln = obj->ClassName();
       cout << kln.Data() ; 
       if( kln == "TObjString" ){      
          cout << ((TObjString*)obj)->GetString() << endl; 
       } else if( kln == "AbtRunInfo" ){       
          ((AbtRunInfo*)obj)->Print() ;
       } else if ( kln == "AbtEvent" ){     
          ((AbtEvent*)obj)->Print() ;
       } else {
          cout << "SKIPPING received obj of class " << kln.Data() << endl ;
       }
   }
   return 0; 
}


int main(int argc, char const * const *argv) 
{
   MQ* q = new MQ();
   //q->Wait( handlebytes , (void*)q );
   delete q ;
   return 0;
}

  

