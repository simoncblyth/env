#define G4DAE_DAYABAY

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETrojanSensDet.hh"

#include "G4SDManager.hh"


#ifdef G4DAE_DAYABAY
#include "Event/SimPmtHit.h"
#include "G4DataHelpers/G4DhHit.h"

namespace DayaBay {
    class SimPmtHit;
}
#endif

using namespace std ;



int main()
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd1 = new G4DAESensDet("DsPmtSensDet");
   sd1->initialize();
   G4DAESensDet* sd2 = new G4DAESensDet("DsRpcSensDet");
   sd2->initialize();

   SDMan->AddNewDetector( sd1 );
   SDMan->AddNewDetector( sd2 );
   // the above is done by GiGa/DetDesc 


   // hmm writing to GDML looses associated SD it seems , so for real

   // need to register the Trojan SD at initialization time,  GiGa hook ?
   // Trojan Horse SD to gain access to HCE via Initialize 
   // 2nd parameter target must match the name of an existing SD 
   //G4SDManager::GetSDMpointer()->AddNewDetector(new G4DAETrojanSensDet("Trojan_DsPmtSensDet", "DsPmtSensDet"));  

   string target = "DsPmtSensDet" ; 

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   chroma->RegisterTrojanSD(target); 


   G4SDManager::GetSDMpointer()->ListTree();


   // below is done by G4 framework and simulation stepping  
   G4HCofThisEvent* HCE = SDMan->PrepareNewEvent();  // calls Initialize for registered SD 

   sd1->ProcessHits(NULL, NULL);
   sd2->ProcessHits(NULL, NULL);




    
   /* 

   // adding extra hits needs access to the tsd
   G4DAETrojanSensDet* TSD = (G4DAETrojanSensDet*)G4SDManager::GetSDMpointer()->FindSensitiveDetector("Trojan_DsPmtSensDet", true); 

   int myints[] = {
                   0x1010101,
                   0x2010101,
                   0x4010101,
                 };

   vector<int> pmtids( myints, myints + sizeof(myints) / sizeof(int) );
   for (vector<int>::iterator it = pmtids.begin(); it != pmtids.end(); ++it)
   {
       int trackid = 1 ; 

       DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
       sphit->setSensDetId(*it); 

       TSD->StoreHit( sphit, trackid );
   }

  */


  G4DAETrojanSensDet* tsd = chroma->GetTrojanSD(target);  
  tsd->AddSomeFakeHits(); 


   // framework calls this
   //sd1->EndOfEvent(HCE);
   //sd2->EndOfEvent(HCE);
   tsd->EndOfEvent(HCE);
}



