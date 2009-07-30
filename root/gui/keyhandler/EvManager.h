#ifndef EVMANAGER_H
#define EVMANAGER_H

#include <TGFrame.h>
#include <RQ_OBJECT.h>

class KeyHandler ;



class EvManager {

   RQ_OBJECT("EvManager")

public:
   EvManager();
   virtual ~EvManager();

   void NextEvent();   // *SIGNAL*
   void PrevEvent();   // *SIGNAL*
   void LoadEvent();   // *SIGNAL*

   Int_t GetEventId();

private:
   Int_t       fEventId   ;
   KeyHandler* fKeyHandler ;       // Handler for key presses used for quick navigation   

public:
   static EvManager* Create();
   ClassDef(EvManager, 0 ) 
};


R__EXTERN EvManager* g_ ;




class KeyHandler : public TGFrame {

public:
   KeyHandler();
   ~KeyHandler();

   Bool_t HandleKey(Event_t *event);    // handler of the key events

   ClassDef(KeyHandler, 0 )  // attempt to adapt ROOTSYS/test/Tetris.cxx key handler
};

#endif 
