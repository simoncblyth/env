#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAECollector.hh"

#include "G4SDManager.hh"
#include <string>

using namespace std ; 


G4DAESensDet* G4DAESensDet::MakeSensDet(const char* name, const char* target )
{
    if ( target == NULL ) return new G4DAESensDet(name, NULL);

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    G4VSensitiveDetector* nameSD = SDMan->FindSensitiveDetector(name, false);
    G4VSensitiveDetector* targetSD = SDMan->FindSensitiveDetector(target, false);

    if( targetSD == NULL || nameSD != NULL)
    {
       cout << "G4DAESensDet::MakeSensDet ERROR failed to make trojan " 
            << " target: " << target 
            << " targetSD: " << targetSD 
            << " name: " << name
            << " nameSD: " << nameSD
            << endl ; 
        return NULL ; 
    } 
    return new G4DAESensDet(name, target);
}

G4DAESensDet::G4DAESensDet(const char* name, const char* target) : G4VSensitiveDetector(name), m_target(target), m_collector(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}

int G4DAESensDet::initialize()
{
   m_collector->DefineCollectionNames(collectionName);
   return 0 ; 
}

void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
    if( m_target == NULL ){
        m_collector->CreateHitCollections( SensitiveDetectorName, hce );
    } else { // trojan 
        m_collector->StealHitCollections( m_target, hce );
    }
}
bool G4DAESensDet::ProcessHits(G4Step* /*step*/, G4TouchableHistory* /*history*/)
{
    assert(0);
    return true ; 
}

void G4DAESensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    m_collector->DumpStatistics(hce);
}

void G4DAESensDet::CollectHits(ChromaPhotonList* cpl, G4DAEGeometry* geometry )
{
   m_collector->CollectHits( cpl, geometry ); 
}


void G4DAESensDet::SetCollector(G4DAECollector* col){
   m_collector = col ; 
}
G4DAECollector* G4DAESensDet::GetCollector(){
   return m_collector ;
}




