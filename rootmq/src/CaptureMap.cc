
#include "CaptureMap.h"
#include "TObject.h"


void CaptureMap::Add(string k , TObject* obj)
{
     Capture c ;
     obj->Print();
     m.insert( Pair(k,c.Gotcha() ));
}


