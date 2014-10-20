
#include "G4DAEChroma/G4DAETrojanSensDet.hh"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include <iostream>
#include <iomanip>


#ifdef G4DAE_DAYABAY
#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"
#endif



using namespace std; 

G4DAETrojanSensDet::G4DAETrojanSensDet(const std::string& name,  const std::string& target) : G4DAESensDet(name), m_target(target)
{
   cout << "G4DAETrojanSensDet::G4DAETrojanSensDet name " << name << " GetName() " << GetName() << "target " <<  GetTargetName() << endl ;
   CheckTarget();
}

G4DAETrojanSensDet::~G4DAETrojanSensDet()
{
}


void G4DAETrojanSensDet::CheckTarget()
{
  G4VSensitiveDetector* tgt = GetTarget();
   if( tgt == NULL ){
      cout << "G4DAETrojanSensDet::G4DAETrojanSensDet WARNING target SD " << GetTargetName() << " not found " << endl ; 
   } else {
      cout <<  "G4DAETrojanSensDet::G4DAETrojanSensDet found target " <<  tgt << " name " << tgt->GetName() << endl ; 
   }
}  


void G4DAETrojanSensDet::Initialize( G4HCofThisEvent* hce )
{
    m_hc.clear();
    CacheHitCollections( GetTargetName(), hce );
}

bool G4DAETrojanSensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
    assert(0); // should never be called
}

void G4DAETrojanSensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    DumpStatistics(hce);
}




std::string G4DAETrojanSensDet::GetTargetName(){
    return m_target ; 
}
G4VSensitiveDetector* G4DAETrojanSensDet::GetTarget()
{
    return G4SDManager::GetSDMpointer()->FindSensitiveDetector(GetTargetName());
}




int G4DAETrojanSensDet::CacheHitCollections(const std::string& name,  G4HCofThisEvent* HCE)
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();

   G4HCtable* hct = SDMan->GetHCtable();
   for(G4int i=0 ; i < hct->entries() ; i++ )
   {
      string sdName = hct->GetSDname(i);  
      string colName = hct->GetHCname(i);  

      if(sdName != name) continue ;

      G4String query = sdName + "/" + colName ; 

      int hcid = hct->GetCollectionID(query);


#ifdef G4DAE_DAYABAY
    
      G4DhHitCollection* hc = (G4DhHitCollection*)HCE->GetHC(hcid); 

      DayaBay::Detector det(colName);
      if(det.bogus()) cout << "WARNING bogus det " << det << endl ;
      //if(det.bogus()) continue ;
      short int detid = det.siteDetPackedData();

      if(m_hc.find(detid) != m_hc.end()) cout << "WARNING : replacing hitcache entry with key " << detid << endl ;
      m_hc[detid] = hc ;

      /*
      cout 
           << " i "       << setw(3)  << i 
           << " sd "      << setw(20) << sdName
           << " hc "      << setw(20) << colName
           << " det "     << setw(20) << det.detName() 
           << " hcid "    << setw(10) << hcid 
           << " detid "   << setw(10) << (void*)detid  
           << " hc "      << setw(10) << hc 
           << endl ;  
      */

#endif


   } 
   return 0;
}


