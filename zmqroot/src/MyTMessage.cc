#include "ZMQRoot/MyTMessage.hh"

#include <stdlib.h>
#include <assert.h>


MyTMessage::MyTMessage( UInt_t what ) : TMessage( what ) { }
MyTMessage::MyTMessage( void *buf, Int_t len) : TMessage( buf, len) { }

TObject* MyTMessage::MyReadObject()
{
   // reconstruct original object 
   assert( this->What() == kMESS_OBJECT ); 
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
   // no need to call base class dtor
}
