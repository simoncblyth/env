#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "Chroma/ChromaPhotonList.hh"

#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

class ITouchableToDetectorElement ;


using namespace std ;
#include <iostream>

#define NOT_NUWA 1
#include "DsChromaRunAction_BeginOfRunAction.icc"


void Mockup_DetDesc_SD()
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd1 = new G4DAESensDet("DsPmtSensDet","");
   sd1->SetCollector(new DybG4DAECollector );  
   sd1->initialize();
   SDMan->AddNewDetector( sd1 );

   G4DAESensDet* sd2 = new G4DAESensDet("DsRpcSensDet","");
   sd2->SetCollector(new DybG4DAECollector );  
   sd2->initialize();
   SDMan->AddNewDetector( sd2 );
}



G4HCofThisEvent* Mockup_NewEvent()
{
   G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 
   return HCE ; 
}


void DsChromaStackAction_ClassifyNewTrack(int pmtid)
{
   // mock OP know the PMT in their destiny 

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   G4DAETransport*   transport = chroma->GetTransport();
   G4DAETransformCache*  cache = chroma->GetTransformCache();

   G4AffineTransform* pg2l = cache->GetSensorTransform(pmtid);
   assert(pg2l);

   G4AffineTransform g2l(*pg2l);
   G4AffineTransform l2g(g2l.Inverse());

   G4ThreeVector lpos(0,0,1500) ; 
   G4ThreeVector ldir(0,0,-1) ;   //  ( lx, ly, lz )  => ( gz, 0.761 gx + 0.6481 gy, -0.64812 gx + 0.761538 gy )  
   G4ThreeVector lpol(0,0,1) ; 

   G4ThreeVector gpos(l2g.TransformPoint(lpos));
   G4ThreeVector gdir(l2g.TransformAxis(ldir));
   G4ThreeVector gpol(l2g.TransformAxis(lpol));

   const float time = 1. ;
   const float wavelength = 550. ;

   transport->CollectPhoton( gpos, gdir, gpol, time, wavelength, pmtid );
}


void DsChromaStackAction_NewStage()
{
   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   G4DAETransport* transport = chroma->GetTransport();

   transport->GetPhotons()->Print();
   transport->GetPhotons()->Details(0);
   std::string digest = transport->GetPhotons()->GetDigest();
   cout << "CPL digest : " << digest << endl ; 

   //transport->GetPhotons()->Save("mock002");  // ldir +y
   //transport->GetPhotons()->Save("mock003");  // ldir +x
   //transport->GetPhotons()->Save("mock004");  // ldir +z
   //transport->GetPhotons()->Save("mock005");  //   lpos (0,0,100) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock006");  //   lpos (0,0,500) ldir (0,0,-1)  try to shoot directly at PMT 
   //transport->GetPhotons()->Save("mock007");  //   lpos (0,0,1500) ldir (0,0,-1)  try to shoot directly at PMT 
   transport->GetPhotons()->Save("mock001");  //   lpos (0,0,1500) ldir (0,0,-1)  try to shoot directly at PMT 

   /*
      g4daeview.sh --load mock002 --nopropagate --geometry-regexp PmtHemiCathode

      udp.py --load mock002 
      udp.py --load mock003 
      udp.py --propagate

   */


   chroma->Propagate(-1); // <1  fakes the propagation, ie just passes all photons off as hits

   /*

      Network setup
      ~~~~~~~~~~~~~~~ 
 
      czmq-
      czmq-broker-local 

      g4daeview.sh --zmqendpoint=tcp://localhost:5002

      OR g4daechroma.sh

      mocknuwa-
      mocknuwa-runenv
      G4DAECHROMA_CLIENT_CONFIG=tcp://localhost:5001 mocknuwa


      From file
      ~~~~~~~~~~

       debug propagation with 

           daedirectpropagation.sh mock001

   */


}


void Mockup_EndEvent(G4HCofThisEvent* HCE )
{
   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->EndOfEvent(HCE);     // G4 calls this
}



int main()
{
    Mockup_DetDesc_SD();

    string transport = "G4DAECHROMA_CLIENT_CONFIG" ;
    string cachekey = "G4DAECHROMA_CACHE_DIR" ;
    string sensdet = "DsPmtSensDet" ;
    ITouchableToDetectorElement* t2de = NULL ;

    DsChromaRunAction_BeginOfRunAction( transport, cachekey, sensdet , t2de , "" );
    G4SDManager::GetSDMpointer()->ListTree();

    G4HCofThisEvent* HCE = Mockup_NewEvent();


    G4DAETransformCache* cache = G4DAEChroma::GetG4DAEChroma()->GetTransformCache();

    // mockup a hit for every PMT    
    for( size_t index = 0 ; index < cache->GetSize() ; ++index )
    {
        if( index % 10 == 0 ) DsChromaStackAction_ClassifyNewTrack(cache->GetKey(index));
    } 
    //DsChromaStackAction_ClassifyNewTrack(cache->GetKey(0));
    //DsChromaStackAction_ClassifyNewTrack(0x1010101);


    DsChromaStackAction_NewStage();

    Mockup_EndEvent(HCE);
}




