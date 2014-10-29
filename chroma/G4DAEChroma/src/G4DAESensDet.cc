#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAECollector.hh"

#include "G4SDManager.hh"

using namespace std ; 


G4DAESensDet* G4DAESensDet::MakeSensDet(const string& name, const string& target )
{
    if ( target.empty() ) return new G4DAESensDet(name, target);

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

G4DAESensDet::G4DAESensDet(const string& name, const string& target) : G4VSensitiveDetector(name), m_target(target), m_collector(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}

void G4DAESensDet::Print()
{
    cout << "G4DAESensDet::Print name " << GetName() << " target [" << m_target << "]" << endl ; 
}


int G4DAESensDet::initialize()
{
   m_collector->DefineCollectionNames(collectionName);
   return 0 ; 
}

void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
    cout << "G4DAESensDet::Initialize hce " << hce << endl ; 
    if( m_target.empty() )
    {
        m_collector->CreateHitCollections( SensitiveDetectorName, hce );
    } 
    else 
    {   
        // trojan
        cout << "G4DAESensDet::Initialize calling StealHitCollections with m_target " << m_target <<  " HCE " << hce << endl ;
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




