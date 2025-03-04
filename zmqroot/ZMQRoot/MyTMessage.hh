#ifndef mytmessage_h
#define mytmessage_h

#include "TMessage.h"
#include "TObject.h"

// http://www.jlab.org/Hall-D/software/wiki/index.php/Serializing_and_deserializing_root_objects

class MyTMessage : public TMessage {

public:
   MyTMessage( UInt_t what = kMESS_ANY );
   MyTMessage(void *buf, Int_t len);

   virtual ~MyTMessage();
   void CopyIntoArray( char* arr );

   TObject* MyReadObject();
   ClassDef( MyTMessage , 1 ) // TMessage with from buffer constructor exposed 
};


#endif
