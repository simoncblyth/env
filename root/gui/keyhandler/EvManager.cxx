
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

EvManager::EvManager()  :
   fEntry(0) ,
   fEntryMin(0),
   fEntryMax(INT_MAX),
   fSource("") 
{
   if (g_ != 0)
      throw("There can be only one!");
   g_ = this;

   fKeyHandler = new KeyHandler ;
}

EvManager::~EvManager(){
   delete fKeyHandler ; fKeyHandler = NULL ;
}


EvManager* EvManager::Create()
{
   if (g_ == 0) g_ = new EvManager();
   return g_ ;
}


void EvManager::Print(Option_t* ) const
{
   cout << IsA()->GetName() << "\t" << GetSource() << "\t" << GetEntry() << "\t" << GetEntryMin() << "\t" << GetEntryMax() << endl ;
}


Int_t EvManager::GetEntry() const { return fEntry ; }
void EvManager::SetEntry(Int_t entry){
   if( entry == fEntry ) return ;
   if( entry > fEntryMax || entry < fEntryMin ){
       Error("EvManager::SetEntry" , Form("Entry %d is not in allowed range " , entry ) );
       return ;
   }
   fEntry = entry ;
   Emit("SetEntry(Int_t)");
}



Int_t EvManager::GetEntryMin() const { return fEntryMin ; }
Int_t EvManager::GetEntryMax() const { return fEntryMax ; }
void EvManager::SetEntryMinMax(Int_t min, Int_t max){
   if( min == fEntryMin && max == fEntryMax ) return ;

   if( fEntry < fEntryMin || fEntry > fEntryMax ){
       Error("EvManager::SetEntryMinMax" , Form("Cannot set range that disallows current entry %d " , fEntry )) ;
       return ;
   }

   fEntryMin = min ;
   fEntryMax = max ;
   Emit("SetEntryMinMax(Int_t,Int_t)");
}


const char* EvManager::GetSource() const { return fSource ; }
void EvManager::SetSource( const char* source ){
   if( source == fSource ) return ;
   fSource = source ;
   Emit("SetSource(char*)");
}



void EvManager::NextEntry(){ 
    SetEntry( GetEntry() + 1 )  ; 
}
void EvManager::PrevEntry(){ 
    SetEntry( GetEntry() - 1 )  ; 
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

  // these 3 work for me ... from OSX 
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
            g_->NextEntry();
            break;
         case kKey_PageUp:
            cout << "Pup" << endl; 
            g_->PrevEntry();
            break;
         case kKey_Space:
            cout << "space" << endl; 
            g_->SetEntry(0);
            break;
         default:
            cout << "default:" << keysym << endl; 
            return kTRUE;
      }
   }
   return kTRUE;
}

