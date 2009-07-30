
#include "EvManager.h"
#include "Riostream.h"

#include "Riostream.h"
#include <TEveManager.h>
#include <TGFrame.h>
#include <KeySymbols.h>

#include "EvManager.h"
#include <RQ_OBJECT.h>

EvManager* g_ = 0;
ClassImp(EvManager);

EvManager::EvManager() {
   if (g_ != 0)
      throw("There can be only one!");
   g_ = this;

   fEventId = 0 ;

   fKeyHandler = new KeyHandler ;
}

EvManager* EvManager::Create(){
   if (g_ == 0){
       g_ = new EvManager();
   }
   return g_ ;
}


Int_t EvManager::GetEventId(){ return fEventId ; }

void EvManager::NextEvent(){
   fEventId += 1 ;
   cout << "EvManager::NextEvent " << fEventId << endl ;
   Emit("NextEvent()");
}
void EvManager::PrevEvent(){
   fEventId -= 1 ;
   cout << "EvManager::PrevEvent " << fEventId << endl ;
   Emit("PrevEvent()");
}
void EvManager::LoadEvent(){
   cout << "EvManager::LoadEvent " << fEventId << endl ;
   Emit("LoadEvent()");
}

EvManager::~EvManager(){
  delete fKeyHandler ; fKeyHandler = NULL ;

}







//  adapted from $ROOTSYS/test/Tetris.cxx

ClassImp(KeyHandler)

KeyHandler::KeyHandler() : TGFrame(gClient->GetRoot(),0,0)
{
   // Key handler constructor.

   TGMainFrame* main_frame = (TGMainFrame*)(gEve->GetBrowser());

   // bind arrow keys and space-bar key
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Up),    0);
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Left),  0);
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Right), 0);
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Down),  0);
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Space), 0);

   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_PageUp), 0);
   main_frame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_PageDown), 0);
}

KeyHandler::~KeyHandler()
{
   // Cleanup key handler.

   // get main frame of Tetris canvas
   TGMainFrame* main_frame = (TGMainFrame*)(gEve->GetBrowser());

   // remove binding of arrow keys and space-bar key
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Up),    0);
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Left),  0);
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Right), 0);
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Down),  0);
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Space), 0);

   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_PageUp), 0);
   main_frame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_PageDown), 0);


   // restore key auto repeat functionality, was turned off in TGMainFrame::HandleKey()
   gVirtualX->SetKeyAutoRepeat(kTRUE);
}


Bool_t KeyHandler::HandleKey(Event_t *event)
{
   // Handle arrow and spacebar keys

   char tmp[2];
   UInt_t keysym;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

   if (event->fType == kGKeyPress) {
      switch ((EKeySym)keysym) {
         case kKey_Left:
            cout << "left" << endl; 
            break;
         case kKey_Right:
            cout << "right" << endl; 
            break;
         case kKey_Down:
            cout << "down" << endl; 
            break;
         case kKey_Up:
            cout << "up" << endl; 
            break;
         case kKey_PageDown:
            cout << "Pdown" << endl; 
            g_->NextEvent();
            break;
         case kKey_PageUp:
            cout << "Pup" << endl; 
            g_->PrevEvent();
            break;
         case kKey_Space:
            cout << "space" << endl; 
            break;
         default:
            cout << "default:" << keysym << endl; 
            return kTRUE;
      }
   }
   return kTRUE;
}

