#ifndef EVMODEL_H
#define EVMODEL_H

#include <RQ_OBJECT.h>
#include <TObject.h>
#include <TString.h>

class EvModel : public TObject {

   RQ_OBJECT("EvModel")

public:
   EvModel();
   virtual ~EvModel();

   void NextEntry();   
   void PrevEntry();    
   void FirstEntry();    
   void LastEntry();    
   void AutoRunUpdated();    
   void AutoEventUpdated();    

   Int_t GetEntry() const;
   void SetEntry(Int_t entry);  // *SIGNAL* 

   Int_t GetDebug() const ;
   void SetDebug(Int_t debug);

   Int_t GetEntryMin() const ;
   Int_t GetEntryMax() const ;
   void SetEntryMinMax(Int_t min, Int_t max); // *SIGNAL*  for an tree with 1000 entries, min/max should be 0, 999

   const char* GetSource() const;
   void SetSource(const char *source);   // *SIGNAL* 
   void RefreshSource();                  // *SIGNAL* 

   const char* GetOther() const;
   void SetOther(const char* other="");  // *SIGNAL*

   void Print(Option_t* option="" ) const;

protected:
   Int_t       fEntry    ;
   Int_t       fEntryMin ;
   Int_t       fEntryMax ;
   Int_t       fDebug ;
   TString     fSource   ;
   TString     fOther    ;

public:
   static EvModel* Create();

   ClassDef(EvModel, 0 ) 
};

R__EXTERN EvModel* g_ ;


#endif 
