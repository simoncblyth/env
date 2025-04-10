#include "G4DAEChroma/G4DAECollector.hh"

#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEHitList.hh"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


G4DAECollector::G4DAECollector() 
#ifdef DEBUG_HITLIST
   : m_hits(NULL)
#endif
{
}
G4DAECollector::~G4DAECollector()
{
#ifdef DEBUG_HITLIST
   delete m_hits ; 
#endif
}


void G4DAECollector::DumpHC( G4VHitsCollection* hc,  int index, int /*detail*/ ) 
{
    string colpath = hc->GetSDname() + "//" + hc->GetName() ;
    cout << " HC " 
         << setw(4) << index << ": " 
         << setw(30) << colpath 
         << " #hits  " << setw(10) << hc->GetSize() 
         << endl;

    hc->PrintAllHits();
}

void G4DAECollector::DumpStatistics( G4HCofThisEvent* hce,  int detail ) 
{
    int ncols = hce->GetNumberOfCollections();
    cout << "G4DAECollector::DumpStatistics "
         << " HCE " << hce
         << " tot collections " << ncols
         << endl ; 

    int tothits = 0;
    for (int index=0; index<ncols; ++index) 
    {
       G4VHitsCollection* hc = hce->GetHC(index);
       if ( hc->GetSize() > 0)
       {
           if ( tothits == 0) cout << endl; 
           DumpHC(hc, index, detail);
       }
       tothits += hc->GetSize() ;
    }
    if ( tothits == 0 ) cout << "G4DAECollector::DumpStatistics WARNING  No hits found in " << ncols << " collections."  << endl;
}


void G4DAECollector::CollectHits( G4DAEArrayHolder* holder, G4DAETransformCache* cache )
{ 
    G4DAEPhotonList* photons = new G4DAEPhotonList(holder);
    std::size_t size = photons->GetCount(); 

#ifdef VERBOSE
    cout << "G4DAECollector::CollectHits size: " << size <<  endl ;   
    photons->Print();
#endif

#ifdef DEBUG_HITLIST
    delete m_hits ;
    m_hits = new G4DAEHitList(size);
#endif

    G4DAEHit hit ;
    for( std::size_t index = 0 ; index < size ; index++ )
    {
        hit.Init( photons, index); 

        G4AffineTransform* transform = ( cache == NULL ) ? NULL :  cache->GetSensorTransform(hit.pmtid) ;
        hit.LocalTransform( transform );  

        this->Collect( hit );

#ifdef DEBUG_HITLIST
        m_hits->AddHit( hit ); 
#endif
    }   
}


#ifdef DEBUG_HITLIST
G4DAEHitList* G4DAECollector::GetHits()
{
   return m_hits ; 
}
#endif

void G4DAECollector::AddSomeFakeHits(const IDVec& sensor_ids)
{
    cout << "G4DAECollector::AddSomeFakeHits" << endl;
    G4DAEHit hit ;
    for (IDVec::const_iterator it = sensor_ids.begin(); it != sensor_ids.end(); ++it)
    {   
        hit.InitFake( *it , 0 ); 
        this->Collect( hit );
    }
}



