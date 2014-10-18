#include "SensDet.h"
#include "G4SDManager.hh"
#include <iostream>
#include <iomanip>

#include "Event/SimPmtHit.h"

#include "G4DataHelpers/G4DhHit.h"

namespace DayaBay {
    class SimPmtHit;
}



using namespace std ;


int main()
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   SensDet* sd = new SensDet("SensDet");
   sd->initialize();
   SDMan->AddNewDetector( sd );
   SDMan->ListTree();
   // registration done by DetDesc/GiGa within real NuWa?
   // needs to be after collectionName inserts

   G4HCofThisEvent* hce = SDMan->PrepareNewEvent();
   //sd->Initialize(hce);  done by PrepareNewEvent it seems

   sd->ProcessHits(NULL, NULL);


   // external access to SensDet hit collections via SDMan inspection
   // lookup on just the collection name assuming no slash in HCname



   typedef std::map<short int,G4DhHitCollection*> ExternalHitCache;
   ExternalHitCache x_hc ;

   G4HCtable* hct = SDMan->GetHCtable();
   for(G4int i=0 ; i < hct->entries() ; i++ )
   {
      G4String colName = hct->GetHCname(i);  
      int hcid = hct->GetCollectionID(colName);
  
      DayaBay::Detector det(colName);

      if(det.bogus()) cout << "WARNING bogus det " << det << endl ;
      //if(det.bogus()) continue ;

      short int detid = det.siteDetPackedData();
      G4DhHitCollection* hc = (G4DhHitCollection*)hce->GetHC(hcid); 
      x_hc[detid] = hc ;
      cout 
           << " i "       << setw(3)  << i 
           << " sd "      << setw(20) << hct->GetSDname(i)
           << " hc "      << setw(20) << colName
           << " det "     << setw(20) << det.detName() 
           << " hcid "    << setw(10) << hcid 
           << " detid "   << setw(10) << (void*)detid  
           << " hc "      << setw(10) << hc 
           << endl ;  
   } 





  int myints[] = {
                   0x1010101,
                   0x2010101,
                   0x4010101,
                 };

  vector<int> pmt( myints, myints + sizeof(myints) / sizeof(int) );
  for (vector<int>::iterator it = pmt.begin(); it != pmt.end(); ++it)
  {
       int trackid = 1 ; 
       int pmtid = *it ; 
       DayaBay::Detector hitdet(*it);

       if(hitdet.bogus()) cout << "WARNING bogus hitdet " << hitdet << endl ;
       short int sdid = hitdet.siteDetPackedData();
       G4DhHitCollection* xhc = x_hc[sdid];

       cout << " pmtid " << setw(10) << (void*)pmtid 
            << " hitdet " << setw(20) << hitdet.detName() 
            << " sdid "   << setw(10) << (void*)sdid 
            << " xhc  "   << setw(10) << xhc 
            << endl ;

       DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
       sphit->setSensDetId(pmtid); 
       xhc->insert(new G4DhHit(sphit,trackid));

  }


   sd->EndOfEvent(hce);
}



