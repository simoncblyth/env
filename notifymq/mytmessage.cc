#include "mytmessage.h"

TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   return this->ReadObject(this->GetClass());
}


