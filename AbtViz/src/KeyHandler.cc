#include "KeyHandler.h"

using namespace std ; 
#include <iostream>

//#include <TEveManager.h>
#include <TGFrame.h>
#include <KeySymbols.h>

#include "EvModel.h"


// DONE : extracating gEve from here ... it brings in a boatload of libs too early 
//        by passing main in with the constructor done in evgui.py 
// 
//  adapted from $ROOTSYS/test/Tetris.cxx
//
// TODO : extracate g_ from here ... by using signals ?
//

ClassImp(KeyHandler)

KeyHandler::KeyHandler(TGMainFrame* main ) : TGFrame(gClient->GetRoot(),0,0)
{
   // Key handler constructor.

   //if(gEve == NULL) return ; 
   //TGMainFrame* main_frame = (TGMainFrame*)(gEve->GetBrowser());

   SetMainFrame( main );
   Bind();
}


void KeyHandler::SetMainFrame( TGMainFrame* main )
{
   fMainFrame = main ;
}

TGMainFrame* KeyHandler::GetMainFrame()
{
    return fMainFrame ;
}


void KeyHandler::Bind()
{

   if(!fMainFrame){
        Printf("KeyHandler::Bind ERROR fMainFrame is NULL \n");
        return ;
   }

   // bind arrow keys and space-bar key
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Up),    0);
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Left),  0);
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Right), 0);
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Down),  0);

  // these 3 work for me ... from OSX 
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_Space), 0);
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_PageUp), 0);
   fMainFrame->BindKey((const TGWindow*)this, gVirtualX->KeysymToKeycode(kKey_PageDown), 0);

}

void KeyHandler::RemoveBind()
{
   if(!fMainFrame){
        Printf("KeyHandler::RemoveBind ERROR fMainFrame is NULL \n");
        return ;
   }

   // remove binding of arrow keys and space-bar key
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Up),    0);
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Left),  0);
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Right), 0);
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Down),  0);

   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_Space), 0);
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_PageUp), 0);
   fMainFrame->RemoveBind(this, gVirtualX->KeysymToKeycode(kKey_PageDown), 0);

}




KeyHandler::~KeyHandler()
{
   // Cleanup key handler.

   //if(gEve == NULL) return ; 
   //TGMainFrame* main_frame = (TGMainFrame*)(gEve->GetBrowser());

   RemoveBind();

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
            //cout << "Pdown" << endl; 
            g_->NextEntry();
            break;
         case kKey_PageUp:
            //cout << "Pup" << endl; 
            g_->PrevEntry();
            break;
         case kKey_Space:
            cout << "space" << endl; 
            g_->FirstEntry();
            break;
         default:
            cout << "default:" << keysym << endl; 
            return kTRUE;
      }
   }
   return kTRUE;
}
