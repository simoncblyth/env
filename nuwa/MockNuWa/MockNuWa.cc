#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransport.hh"

#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"

#include "Chroma/ChromaPhotonList.hh"

#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

using namespace std ;
#include <iostream>

int main()
{
   ////////// Mockup SD matching NuWa/GiGa/DetDesc ///////////

   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd1 = new G4DAESensDet("DsPmtSensDet");
   sd1->SetCollector(new DybG4DAECollector );  
   sd1->initialize();
   SDMan->AddNewDetector( sd1 );

   G4DAESensDet* sd2 = new G4DAESensDet("DsRpcSensDet");
   sd2->SetCollector(new DybG4DAECollector );  
   sd2->initialize();
   SDMan->AddNewDetector( sd2 );


   /////////// DsChromaRunAction::BeginOfRun //////////////////////////
   ///
   ///  * configure G4DAEChroma singleton
   ///  * add trojan SD, providing backdoor for adding "GPU" hits 
   ///
   /////////////////////////////////////////////////////////////////////

   const char* transport = "G4DAECHROMA_CLIENT_CONFIG" ;
   const char* sensdet = "DsPmtSensDet" ;
   const char* geometry = "DAE_NAME_DYB_GDML" ;

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   chroma->Configure( transport, sensdet, geometry );

   DybG4DAECollector* collector = new DybG4DAECollector ;
   G4DAESensDet* sd = chroma->GetSensDet();  

   if( sd == NULL ){
      cout << "no SD named " << sensdet << endl ; 
      return 1 ; 
   } 

   sd->SetCollector(collector); 
   sd->initialize();
   SDMan->AddNewDetector( sd );
   G4SDManager::GetSDMpointer()->ListTree();


   /////////////  G4EventManager? ////////////////////////////

   G4HCofThisEvent* HCE = SDMan->PrepareNewEvent();  // calls Initialize for registered SD 

   /////////////  G4Stepping /////////////////////

   //sd1->ProcessHits(NULL, NULL);
   //sd2->ProcessHits(NULL, NULL);

   ////////////  DsChromaStackAction::ClassifyNewTrack (maybe in processes in future) //////////////

   const G4ThreeVector pos ;
   const G4ThreeVector dir ;
   const G4ThreeVector pol ;
   const float time = 1. ;
   const float wavelength = 550. ;

   G4DAETransport* tra = chroma->GetTransport();

   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x1010101 );
   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x2010101 );
   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x4010101 );
   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x1010101 );
   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x2010101 );
   tra->CollectPhoton( pos, dir, pol, time, wavelength, 0x4010101 );

   tra->GetPhotons()->Print();
   tra->GetPhotons()->Details(0);

   ////////////  DsChromaStackAction::NewStage (maybe in processes in future) //////////////
   
   chroma->Propagate(-1); // -ve fakes the transport 



   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->EndOfEvent(HCE);     // G4 calls this
}








void add_fakehits()
{
   std::size_t ids[] = { 
                     0x1010101,
                     0x2010101,
                     0x4010101,
                   };  

   G4DAECollector::IDVec pmtids( ids, ids + sizeof(ids)/sizeof(std::size_t) );
   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->GetCollector()->AddSomeFakeHits(pmtids); 
}

