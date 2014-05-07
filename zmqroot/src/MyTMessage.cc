#include "MyTMessage.hh"
#include <stdlib.h>


TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   return this->ReadObject(this->GetClass());
}


void MyTMessage::CopyIntoArray( char* arr )
{
  // adapted from Chromaserver
  //
  // Copy the serialized buffer from a TMessage into char* arr, 
  // If we return a char*, PyROOT casts it to a str
  // and cuts it off at the first null character.

    memcpy(arr, this->Buffer(), this->Length());
}



MyTMessage::~MyTMessage()
{
}
