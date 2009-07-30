

#include <TGFrame.h>

//  KeyHandler = virtual frame
//  used to catch and handle key events in Tetris canvas
///////////////////////////////////////////////////////////////////
class KeyHandler : public TGFrame {

public:
   KeyHandler();
   ~KeyHandler();

   Bool_t HandleKey(Event_t *event);    // handler of the key events


   ClassDef(KeyHandler, 0 )  // attempt to adapt ROOTSYS/test/Tetris.cxx key handler
};



