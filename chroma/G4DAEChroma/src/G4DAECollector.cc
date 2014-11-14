#include "G4DAEChroma/G4DAECollector.hh"

#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4DAEChroma/G4DAEHitList.hh"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


G4DAECollector::G4DAECollector() : m_hits(NULL)
{
}
G4DAECollector::~G4DAECollector()
{
   delete m_hits ; 
}



void G4DAECollector::DumpStatistics( G4HCofThisEvent* hce ) 
{
    int ncols = hce->GetNumberOfCollections();
    cout << "G4DAECollector::DumpStatistics "
         << " HCE " << hce
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
    if ( tothits == 0 ) cout << "G4DAECollector::DumpStatistics WARNING  No hits found in " << ncols << " collections."  << endl;
}


void G4DAECollector::CollectHits( G4DAEPhotons* photons, G4DAETransformCache* cache )
{ 
    std::size_t size = photons->GetCount(); 

#ifdef VERBOSE
    cout << "G4DAECollector::CollectHits size: " << size <<  endl ;   
    photons->Print();
#endif

    delete m_hits ;
    m_hits = new G4DAEHitList(size);

    G4DAEHit hit ;
    for( std::size_t index = 0 ; index < size ; index++ )
    {
        hit.Init( photons, index); 

        G4AffineTransform* transform = ( cache == NULL ) ? NULL :  cache->GetSensorTransform(hit.pmtid) ;
        hit.LocalTransform( transform );  

        this->Collect( hit );

        m_hits->AddHit( hit ); 
    }   
}


G4DAEHitList* G4DAECollector::GetHits()
{
   return m_hits ; 
}

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



