#ifndef mytmessage_h
#define mytmessage_h

#include "TMessage.h"
#include "TObject.h"

// http://www.jlab.org/Hall-D/software/wiki/index.php/Serializing_and_deserializing_root_objects

class MyTMessage : public TMessage {

public:
   MyTMessage(void *buf, Int_t len) : TMessage(buf, len) { }
   virtual ~MyTMessage(){}
   TObject* MyReadObject();
   ClassDef( MyTMessage , 1 ) // TMessage with from buffer constructor exposed 
};


#endif
