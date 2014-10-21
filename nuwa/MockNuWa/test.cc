#define G4DAE_DAYABAY

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransport.hh"

#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETrojanSensDet.hh"

#include "G4SDManager.hh"

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



   G4DAETransport* tra =  G4DAETransport::MakeTransport("G4DAECHROMA_CLIENT_CONFIG");
   G4DAEGeometry*  geo =  G4DAEGeometry::LoadFromGDML("DAE_NAME_DYB_GDML");
   G4DAETrojanSensDet* tsd = G4DAETrojanSensDet::MakeTrojanSensDet("DsPmtSensDet", geo ); // registration done inside

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

   chroma->SetSensDet( tsd ); 
   chroma->SetGeometry( geo );    // duplication: also in SD, remove ?
   chroma->SetTransport( tra );





   G4SDManager::GetSDMpointer()->ListTree();

   // below is done by G4 framework and simulation stepping  
   G4HCofThisEvent* HCE = SDMan->PrepareNewEvent();  // calls Initialize for registered SD 

   sd1->ProcessHits(NULL, NULL);
   sd2->ProcessHits(NULL, NULL);


    
   /* 
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


  //tsd->AddSomeFakeHits(); 


   // framework calls this
   //sd1->EndOfEvent(HCE);
   //sd2->EndOfEvent(HCE);
   tsd->EndOfEvent(HCE);
}



