#include "MyTMessage.h"

ClassImp(MyTMessage)

TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   return this->ReadObject(this->GetClass());
}

MyTMessage::~MyTMessage()
{
}
