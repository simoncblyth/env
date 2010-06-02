#ifndef KEYHANDLER_H 
#define KEYHANDLER_H

#include <TGFrame.h>

class TGMainFrame ;

class KeyHandler : public TGFrame {

private:
   TGMainFrame* fMainFrame ;

public:
   KeyHandler(TGMainFrame* main);
   ~KeyHandler();

   void SetMainFrame(TGMainFrame* main);
   TGMainFrame* GetMainFrame();
   void Bind();
   void RemoveBind();

   Bool_t HandleKey(Event_t *event);    // handler of the key events

   ClassDef(KeyHandler, 0 )  // attempt to adapt ROOTSYS/test/Tetris.cxx key handler
};

#endif 
