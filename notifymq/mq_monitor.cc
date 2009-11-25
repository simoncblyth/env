
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
   while(!gMQ->IsMonitorFinished()){
      if(gMQ->IsBytesUpdated()){
         cout << "gMQ BytesUpdated " << endl;  
         TObject* obj = gMQ->ConstructObject();
         if(obj) obj->Print();  
      } 
      gSystem->Sleep(100);
      gSystem->ProcessEvents();
   }
   return 0;
}


