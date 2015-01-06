#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"


#include "G4SDManager.hh"

using namespace std ; 


void G4DAESensDet::MockupSD(const char* name, G4DAECollector* collector)
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd = new G4DAESensDet(name,"");
   sd->SetCollector(collector);  
   sd->initialize();
   SDMan->AddNewDetector( sd );
}


G4DAESensDet* G4DAESensDet::MakeTrojanSensDet(const string& target, G4DAECollector* collector)
{
    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    G4VSensitiveDetector* targetSD = SDMan->FindSensitiveDetector(target, false);
    if( targetSD == NULL)
    {
        cout << "G4DAESensDet::MakeTrojanSensDet ERROR no such target SD exists  " << target << endl ;  
        return NULL ; 
    } 
    // this one has access to the standard Geant4 hit collections
    string name = "trojan_" ;
    name += target ;

    printf("G4DAESensDet::MakeTrojanSensDet : stealing hit collections from target %s into %s \n", target.c_str(), name.c_str() );

    G4DAESensDet* sensdet = new G4DAESensDet(name, target);
    sensdet->SetCollector(collector); 
    SDMan->AddNewDetector( sensdet );

    cout << "G4DAESensDet::MakeTrojanSensDet AddNewDetector [" << sensdet->GetName() << "]" << endl ; 
    return sensdet ;    
}


G4DAESensDet* G4DAESensDet::MakeChromaSensDet(const string& target, G4DAECollector* collector)
{
    string name = "chroma_" ;
    name += target ;

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    G4VSensitiveDetector* nameSD = SDMan->FindSensitiveDetector(name, false);
    if( nameSD != NULL)
    {
        cout << "G4DAESensDet::MakeChromaSensDet ERROR SD exists already  " << name << endl ;  
        return NULL ; 
    } 

    printf("G4DAESensDet::MakeChromaSensDet : %s \n", name.c_str() );

    G4DAESensDet* sensdet = new G4DAESensDet(name, "");
    sensdet->SetCollector(collector); 
    sensdet->initialize();

    SDMan->AddNewDetector( sensdet );

    cout << "G4DAESensDet::MakeChromaSensDet AddNewDetector [" << sensdet->GetName() << "]" << endl ; 
    return sensdet ;    
}









G4DAESensDet::G4DAESensDet(const string& name, const string& target) : G4VSensitiveDetector(name), m_target(target), m_collector(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}

void G4DAESensDet::Print(const char* msg) const
{
    cout << msg 
         << " name " << GetName() << " target [" << m_target << "]" << endl ; 
}


int G4DAESensDet::initialize()
{
   m_collector->DefineCollectionNames(collectionName);
   return 0 ; 
}

void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
#ifdef VERBOSE
    cout << "G4DAESensDet::Initialize hce " << hce << endl ; 
#endif
    if( m_target.empty() )
    {
        m_collector->CreateHitCollections( SensitiveDetectorName, hce );
    } 
    else 
    {   
        // trojan
#ifdef VERBOSE
        cout << "G4DAESensDet::Initialize calling StealHitCollections with m_target " << m_target <<  " HCE " << hce << endl ;
#endif
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
    cout << "G4DAESensDet::EndOfEvent " << hce << endl ;
#ifdef VERBOSE
    m_collector->DumpStatistics(hce);
#endif
}

void G4DAESensDet::CollectHits(G4DAEPhotonList* photons, G4DAETransformCache* cache )
{
   // in 
   m_collector->CollectHits( photons, cache ); 
}

void G4DAESensDet::PopulatePmtHitList(G4DAEPmtHitList* pmthits)
{
   // out 
   m_collector->PopulatePmtHitList(pmthits);
}


void G4DAESensDet::SetCollector(G4DAECollector* col){
   m_collector = col ; 
}

G4DAECollector* G4DAESensDet::GetCollector(){
   return m_collector ;
}




