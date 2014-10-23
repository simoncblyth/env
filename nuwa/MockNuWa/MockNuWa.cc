#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransport.hh"

#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"

#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

using namespace std ;


//
// SetDetector and G4DAEDayabay are mis-nomers, 
// rename to SetCollector G4DAEDayabayCollector
// this however needs to be done externally 
// as its in the Collector that detector specifics are contained
//

int main()
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd1 = new G4DAESensDet("DsPmtSensDet");
   sd1->SetCollector(new DsChromaG4DAECollector );  
   sd1->initialize();
   SDMan->AddNewDetector( sd1 );

   G4DAESensDet* sd2 = new G4DAESensDet("DsRpcSensDet");
   sd2->SetCollector(new DsChromaG4DAECollector );  
   sd2->initialize();
   SDMan->AddNewDetector( sd2 );

   // **the above is just for Mockup of what is done by GiGa/DetDesc** 


   const char* transport = "G4DAECHROMA_CLIENT_CONFIG" ;
   const char* sensdet = "DsPmtSensDet" ;
   const char* geometry = "DAE_NAME_DYB_GDML" ;

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   chroma->Configure( transport, sensdet, geometry );




   DybG4DAECollector* col = new DybG4DAECollector ;
   G4DAESensDet* sd = chroma->GetSensDet();
   sd->SetCollector(col); 
   sd->initialize();

   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   SDMan->AddNewDetector( sd );







   G4SDManager::GetSDMpointer()->ListTree();


   // below is done by G4 framework and simulation stepping  
   G4HCofThisEvent* HCE = SDMan->PrepareNewEvent();  // calls Initialize for registered SD 

   //sd1->ProcessHits(NULL, NULL);
   //sd2->ProcessHits(NULL, NULL);


   std::size_t ids[] = { 
                     0x1010101,
                     0x2010101,
                     0x4010101,
                   };  

   G4DAECollector::IDVec pmtids( ids, ids + sizeof(ids)/sizeof(std::size_t) );

   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->GetCollector()->AddSomeFakeHits(pmtids); 


   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->EndOfEvent(HCE);     // G4 calls this
}



