
#include "EvModel.h"
#include "Riostream.h"
#include "TClass.h"


EvModel* g_ = 0;
ClassImp(EvModel);

EvModel::EvModel()  :
   fEntry(-1) ,
   fEntryMin(0),
   fEntryMax(INT_MAX),
   fDebug(0),
   fSource("")
{
   if (g_ != 0)
      throw("There can be only one!");
   g_ = this;

 
}

EvModel::~EvModel(){
}



EvModel* EvModel::Create()
{
   if (g_ == 0) g_ = new EvModel();
   return g_ ;
}


void EvModel::Print(Option_t* ) const
{
   cout << IsA()->GetName() << "\t" << GetSource() << "\t" << GetEntry() << "\t" << GetEntryMin() << "\t" << GetEntryMax() << endl ;
}

Int_t EvModel::GetDebug() const { return fDebug ; }
void EvModel::SetDebug(Int_t debug)
{
    fDebug = debug ;
}


Int_t EvModel::GetEntry() const { return fEntry ; }
void EvModel::SetEntry(Int_t entry){

   if( entry < 0 ) entry = fEntryMax + entry + 1 ;         // argument -1 corresponds to the last entry 
   if( entry == fEntry && entry != fEntryMax ) return ;    // when online entry stays at entrymax, but need the signal 
   if( entry > fEntryMax || entry < fEntryMin ){
       Error("EvModel::SetEntry" , Form("Entry %d is not in allowed range " , entry ) );
       return ;
   }
   fEntry = entry ;
   Emit("SetEntry(Int_t)");
}



Int_t EvModel::GetEntryMin() const { return fEntryMin ; }
Int_t EvModel::GetEntryMax() const { return fEntryMax ; }
void EvModel::SetEntryMinMax(Int_t min, Int_t max){
   if( min == fEntryMin && max == fEntryMax ) return ;

   // entry -1 is special cased, as that is how a new EvModel is initialized
   if(( fEntry < fEntryMin || fEntry > fEntryMax ) && fEntry != -1  ){
       Error("EvModel::SetEntryMinMax" , Form("Cannot set range that disallows current entry %d " , fEntry )) ;
       return ;
   }

   fEntryMin = min ;
   fEntryMax = max ;

   // for initial setting, adopt the minimum 
   if( fEntry == -1 ) SetEntry( fEntryMin ) ;

   Emit("SetEntryMinMax(Int_t,Int_t)");
}


const char* EvModel::GetSource() const { return fSource ; }
void EvModel::SetSource( const char* source )
{
   if( source == fSource ) return ;
   fSource = source ;
   Emit("SetSource(char*)");
}

void EvModel::RefreshSource()
{
   if( fSource == "" ) return ;
   Emit("RefreshSource()");
}


void EvModel::AutoRunUpdated()
{
   if(fDebug > 0) cout << "EvModel::AutoRunUpdated NOP " << endl ;
   //SetEntry( -1 );
}
void EvModel::AutoEventUpdated()
{
   if(fDebug > 0) cout << "EvModel::AutoEventUpdated " << endl ;
   SetEntry( -1 );
}


void EvModel::NextEntry()
{ 
    SetEntry( GetEntry() + 1 )  ; 
}
void EvModel::PrevEntry()
{ 
    SetEntry( GetEntry() - 1 )  ; 
}
void EvModel::FirstEntry()
{ 
    SetEntry( GetEntryMin() )  ; 
}
void EvModel::LastEntry()
{ 
    SetEntry( GetEntryMax() )  ; 
}



