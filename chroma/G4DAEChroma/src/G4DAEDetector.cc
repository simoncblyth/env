#include "G4DAEChroma/G4DAEDetector.hh"

#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


void G4DAEDetector::DumpStatistics( G4HCofThisEvent* hce ) 
{
    int ncols = hce->GetNumberOfCollections();
    cout << "G4DAEDetector::DumpStatistics "
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
    if ( tothits == 0 ) cout << "G4DAEDetector::DumpStatistics WARNING  No hits found in " << ncols << " collections."  << endl;
}


void G4DAEDetector::CollectHits( ChromaPhotonList* cpl, G4DAEGeometry* geometry )
{ 
    cout << "G4DAEDetector::CollectHits " <<  endl ;   
    cpl->Print();
    std::size_t size = cpl->GetSize(); 

    G4DAEHit hit ;
    for( std::size_t index = 0 ; index < size ; index++ )
    {
         hit.Init( cpl, index); 
         G4AffineTransform& trans = geometry->GetNodeTransform(hit.volumeindex) ;
         hit.LocalTransform( trans );  

         this->Collect( hit );
    }   
}

