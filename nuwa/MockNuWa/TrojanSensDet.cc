
#include "TrojanSensDet.h"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"

#include "Event/SimPmtHit.h"
#include "Conventions/Detectors.h"

#include <iostream>
#include <iomanip>


using namespace std; 

TrojanSensDet::TrojanSensDet(const std::string& name,  const std::string& target) : G4VSensitiveDetector(name), m_target(target)
{
   cout << "TrojanSensDet::TrojanSensDet name " << name << " GetName() " << GetName() << "target " <<  GetTargetName() << endl ;
}

TrojanSensDet::~TrojanSensDet()
{
}



void TrojanSensDet::Initialize( G4HCofThisEvent* hce )
{
    m_hc.clear();
    CacheHitCollections( GetTargetName(), hce );
}

bool TrojanSensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
    assert(0); // should never be called
}

void TrojanSensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    DumpStatistics(hce);
}




std::string TrojanSensDet::GetTargetName(){
    return m_target ; 
}



int TrojanSensDet::CacheHitCollections(const std::string& name,  G4HCofThisEvent* HCE)
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
   } 
   return 0;
}





void TrojanSensDet::StoreHit(DayaBay::SimPmtHit* sphit, int trackid)
{
    int pmtid = sphit->sensDetId();

    DayaBay::Detector det(pmtid);

    if(det.bogus()) cout << "TrojanSensDet::StoreHit WARNING bogus det " << det << endl ;

    short int sdid = det.siteDetPackedData();
    G4DhHitCollection* hc = m_hc[sdid];

    cout << "TrojanSensDet::StoreHit "
         << " pmtid " << setw(10) << (void*)pmtid 
         << " det "   << setw(20) << det.detName() 
         << " sdid "  << setw(10) << (void*)sdid 
         << " hc  "   << setw(10) << hc 
         << endl ;

    hc->insert(new G4DhHit(sphit,trackid));

}



void TrojanSensDet::DumpStatistics( G4HCofThisEvent* hce ) 
{
    cout << "TrojanSensDet::DumpStatistics HCE Cache has " << m_hc.size() << " collections" << endl ; 

    int ncols = hce->GetNumberOfCollections();
    cout << "SensDet EndOfEvent " << ncols << " collections.";

    int tothits = 0;
    for (int ind=0; ind<ncols; ++ind) {
      G4VHitsCollection* hc = hce->GetHC(ind);
      if ( hc->GetSize() > 0)
      {
          if ( tothits == 0) cout << endl; 
          cout << ind << ": " 
               << hc->GetSDname() << "//" << hc->GetName() << " has " 
               << hc->GetSize() << " hits" << endl; 
      }
      tothits += hc->GetSize() ;
    }
    if ( tothits == 0 ) cout << " No hits found in " << ncols << " collections."  << endl;

}



