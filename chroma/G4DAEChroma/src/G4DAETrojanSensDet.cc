
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


G4DAETrojanSensDet* G4DAETrojanSensDet::MakeTrojanSensDet(const std::string& target, G4DAEGeometry* geometry)
{
    G4SDManager* SDMan = G4SDManager::GetSDMpointer();

    G4VSensitiveDetector* targetSD = SDMan->FindSensitiveDetector(target, false);
    if( targetSD == NULL ){
        cout << "G4DAEChroma::MakeTrojanSensDet ERROR there is no SD called  " << target << endl ;
        return NULL ; 
    }

    string trojan = "Trojan_" + target ;
    G4DAETrojanSensDet* trojanSD = (G4DAETrojanSensDet*)SDMan->FindSensitiveDetector(trojan, false);

    if( trojanSD != NULL )
    {
        cout << "G4DAEChroma::MakeTrojanSensDet WARNING there is already a trojanSD called  " << trojan << " SKIPPING " <<  endl ;
    } 
    else 
    {
        trojanSD = new G4DAETrojanSensDet(trojan, target);
        trojanSD->SetGeometry(geometry);
        SDMan->AddNewDetector(trojanSD);
    }
    return trojanSD ; 
}



G4DAETrojanSensDet* G4DAETrojanSensDet::GetTrojanSensDet(const std::string& target)
{
    string trojan = "Trojan_" + target ;
    return (G4DAETrojanSensDet*)G4SDManager::GetSDMpointer()->FindSensitiveDetector(trojan, true); 
}




G4DAETrojanSensDet::G4DAETrojanSensDet(const std::string& name,  const std::string& target) : G4DAESensDet(name), m_target(target)
{
}

G4DAETrojanSensDet::~G4DAETrojanSensDet()
{
}





void G4DAETrojanSensDet::Initialize( G4HCofThisEvent* hce )
{
    StealHitCollections( m_target, hce );
}

bool G4DAETrojanSensDet::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
    assert(0); // should never be called
}

void G4DAETrojanSensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    DumpStatistics(hce);
}





int G4DAETrojanSensDet::StealHitCollections(const std::string& target,  G4HCofThisEvent* HCE)
{
   /*
   Summary: this steals HCE hit collection pointers of target SD

   For entries in HCtable with SDname matching the argument, 
   obtain hcid and corresponding HC. 
   Cache HC pointers into m_hc keyed by site-detector short int, 
   obtained by DayaBay::Detector interpretation of the collection name.   

   NB this relies on the `G4DAETrojanSensDet::Initialize( G4HCofThisEvent* hce )`
   being called after that of the target SD otherwise will fail to access HC.

   As a result of this access to targetted hit collections of the event
   hits can be added outside of the normal ProcessHits machinery using 
   hit collection methods provided by the `G4DAESensDet` base class.

   */ 

   m_hc.clear();
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();

   G4HCtable* hct = SDMan->GetHCtable();
   for(G4int i=0 ; i < hct->entries() ; i++ )
   {
      string sdName = hct->GetSDname(i);  
      string colName = hct->GetHCname(i);  

      if(sdName != target) continue ;

      G4String query = sdName + "/" + colName ; 

      int hcid = hct->GetCollectionID(query);


#ifdef G4DAE_DAYABAY
    
      G4DhHitCollection* hc = (G4DhHitCollection*)HCE->GetHC(hcid); 

      DayaBay::Detector det(colName);
      if(det.bogus()) cout << "G4DAETrojanSensDet::StealHitCollections : WARNING bogus det " << det << endl ;
      //if(det.bogus()) continue ;
      short int detid = det.siteDetPackedData();

      if(m_hc.find(detid) != m_hc.end()) cout << "G4DAETrojanSensDet::StealHitCollections : WARNING : replacing hitcache entry with key " << detid << endl ;
      m_hc[detid] = hc ;
#endif

   } 


   cout << "G4DAETrojanSensDet::StealHitCollections "
        << " HCE " << HCE
        << " target " << target 
        << " #col " << m_hc.size()
        << endl ; 


   return 0;
}


