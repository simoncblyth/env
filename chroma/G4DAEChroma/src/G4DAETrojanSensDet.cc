
#include "G4DAEChroma/G4DAETrojanSensDet.hh"
#include "G4DAEChroma/G4DAEDetector.hh"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include <iostream>
#include <iomanip>


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

    cout << "G4DAETrojanSensDet::MakeTrojanSensDet SDMan->ListTree() " << endl ;
    SDMan->ListTree();
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
    m_detector->StealHitCollections( m_target.c_str(), hce );
}

bool G4DAETrojanSensDet::ProcessHits(G4Step* /*step*/, G4TouchableHistory* /*history*/)
{
    assert(0); // should never be called
}

void G4DAETrojanSensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    m_detector->DumpStatistics(hce);
}






