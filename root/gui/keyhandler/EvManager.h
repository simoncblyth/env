#ifndef EVMANAGER_H
#define EVMANAGER_H

#include <TGFrame.h>
#include <RQ_OBJECT.h>
#include <TObject.h>
#include <TString.h>

class KeyHandler ;



class EvManager : public TObject {

   RQ_OBJECT("EvManager")

public:
   EvManager();
   virtual ~EvManager();

   void NextEntry();   
   void PrevEntry();    
   void FirstEntry();    
   void LastEntry();    

   Int_t GetEntry() const;
   void SetEntry(Int_t entry);  // *SIGNAL* 

   Int_t GetEntryMin() const ;
   Int_t GetEntryMax() const ;
   void SetEntryMinMax(Int_t min, Int_t max); // *SIGNAL* 

   const char* GetSource() const;
   void SetSource(const char *source);   // *SIGNAL* 

   void Print(Option_t* option="" ) const;

protected:
   Int_t       fEntry    ;
   Int_t       fEntryMin ;
   Int_t       fEntryMax ;
   TString     fSource   ;

private:
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
