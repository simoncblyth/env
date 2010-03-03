
#include <iostream>
using namespace std ;

#include "MQ.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TObject.h"
#include "TString.h"

int main(int argc, char const * const *argv) 
{
   gMQ = MQ::Create(kTRUE);  // start the monitor thread 
   //const char* key = "default.routingkey" ;
   const char* key = "other.routingkey" ;
   cout << "---- mq_monitor.cc : started looking for messages with key " << key  << endl ;
   while(gMQ->IsMonitorRunning()){
         if(gMQ->IsUpdated(key)){
             TObject* obj = gMQ->Get( key, 0 );
             if(obj) obj->Print();  
         }
         gSystem->Sleep(1000);
         gSystem->ProcessEvents();
   }
 

   return 0;
}


