#include "MyTMessage.hh"
#include <stdlib.h>


TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   return this->ReadObject(this->GetClass());
}


void MyTMessage::SerializeIntoArray( char* msg )
{
   // adapted from Chromaserver

  // Copy the serialized buffer from a TBufferFile into char* msg, which is
  // really a Python string. If we return a char*, PyROOT casts it to a str
  // and cuts it off at the first null character.

    memcpy(msg, this->Buffer(), this->Length());
}



MyTMessage::~MyTMessage()
{
}
