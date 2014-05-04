#include "MyTMessage.hh"


TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   return this->ReadObject(this->GetClass());
}

MyTMessage::~MyTMessage()
{
}
