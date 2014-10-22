
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "Chroma/ChromaPhotonList.hh"  
#include "G4DAEChroma/G4DAEHit.hh"
#include "G4DAEChroma/G4DAEDetector.hh"

#include "G4SDManager.hh"

using namespace std; 


G4DAESensDet::G4DAESensDet(const std::string& name) : G4VSensitiveDetector(name), m_geometry(0)
{
}

G4DAESensDet::~G4DAESensDet()
{
}


int G4DAESensDet::initialize()
{
   DefineCollectionNames();
   return 0 ; 
}
void G4DAESensDet::Initialize( G4HCofThisEvent* hce )
{
    CreateHitCollections( hce );
}
bool G4DAESensDet::ProcessHits(G4Step* /*step*/, G4TouchableHistory* /*history*/)
{
    //assert(0);
   int pmtid = 0x1010101 ;
   int trackid = 101 ;
#ifdef G4DAE_DAYABAY
   DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
   sphit->setSensDetId(pmtid); 
   this->StoreHit( sphit, trackid);
#endif
   return true ; 
}






void G4DAESensDet::EndOfEvent( G4HCofThisEvent* hce ) 
{
    DumpStatistics(hce);
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


void G4DAESensDet::CreateHitCollections( G4HCofThisEvent* hce )
{
    m_detector->CreateHitCollections( SensitiveDetectorName, hce );
}




void G4DAESensDet::CollectHits(ChromaPhotonList* cpl)
{
    cout << "G4DAESensDet::CollectHits " <<  endl ;   
    cpl->Print();
    std::size_t size = cpl->GetSize(); 

    G4DAEHit hit ;
    for( std::size_t index = 0 ; index < size ; index++ )
    {
         hit.Init( cpl, index); 
         G4AffineTransform& trans = m_geometry->GetNodeTransform(hit.volumeindex) ;
         hit.LocalTransform( trans );  
         fDetector->CollectOneHit( hit );
    }   
}




void G4DAESensDet::AddSomeFakeHits()
{
    int myints[] = { 
                   0x1010101,
                   0x2010101,
                   0x4010101,
                 };  

    vector<int> pmtids( myints, myints + sizeof(myints) / sizeof(int) );
    for (vector<int>::iterator it = pmtids.begin(); it != pmtids.end(); ++it)
    {   
        int trackid = 1 ; 

#ifdef G4DAE_DAYABAY
        DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
        sphit->setSensDetId(*it); 
        StoreHit( sphit, trackid );
#endif



    }   
}








void G4DAESensDet::DumpStatistics( G4HCofThisEvent* hce ) 
{
    int ncols = hce->GetNumberOfCollections();
    cout << "G4DAESensDet::DumpStatistics "
         << " HCE " << hce
         << " cached HC " << m_hc.size() 
         << " tot collections " << ncols
         << endl ; 

    int tothits = 0;
    for (int ind=0; ind<ncols; ++ind) 
    {
        G4VHitsCollection* hc = hce->GetHC(ind);
       if ( hc->GetSize() > 0)
       {
           string colpath = hc->GetSDname() + "//" + hc->GetName() ;

           if ( tothits == 0) cout << endl; 
           cout << " col# " << setw(4) << ind << ": " 
                << setw(30) << colpath 
                << " #hits  " << setw(10) << hc->GetSize() 
                << endl; 
       }
       tothits += hc->GetSize() ;
    }
    if ( tothits == 0 ) cout << "G4DAESensDet::DumpStatistics WARNING  No hits found in " << ncols << " collections."  << endl;
}



