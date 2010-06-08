
#include "CaptureMap.h"
#include "Capture.h"


void CaptureMap::Add(string k , string v )
{
   /*
     Capture c ;
     obj->Print();
     m.insert( Pair(k,c.Gotcha() ));
   */
     m.insert( Pair(k,v ));
}


