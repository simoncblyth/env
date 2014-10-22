#include "G4DAEChroma/G4DAESensDet.hh"
//#include "G4DAEChroma/G4DAEGeometry.hh"
//#include "Chroma/ChromaPhotonList.hh"  
#include "G4DAEChroma/G4DAEDetector.hh"

using namespace std; 


G4DAESensDet::G4DAESensDet(const std::string& name) : G4VSensitiveDetector(name), m_geometry(0), m_detector(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}


int G4DAESensDet::initialize()
{
   m_detector->DefineCollectionNames(collectionName);
   return 0 ; 
}
void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
    m_detector->CreateHitCollections( SensitiveDetectorName, hce );
}
bool G4DAESensDet::ProcessHits(G4Step* /*step*/, G4TouchableHistory* /*history*/)
{
    assert(0);
    return true ; 
}
void G4DAESensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    m_detector->DumpStatistics(hce);
}
void G4DAESensDet::CollectHits(ChromaPhotonList* cpl, G4DAEGeometry* geometry )
{
   m_detector->CollectHits( cpl, geometry ); 
}





void G4DAESensDet::SetGeometry(G4DAEGeometry* geo){
   m_geometry = geo ; 
}
G4DAEGeometry* G4DAESensDet::GetGeometry(){
   return m_geometry ;
}

void G4DAESensDet::SetDetector(G4DAEDetector* det){
   m_detector = det ; 
}
G4DAEDetector* G4DAESensDet::GetDetector(){
   return m_detector ;
}




