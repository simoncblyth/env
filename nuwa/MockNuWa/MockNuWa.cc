#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "Chroma/ChromaPhotonList.hh"

#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

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
   G4ThreeVector lpos(0,0,0) ; 
   G4ThreeVector gpos(l2g.TransformPoint(lpos));

   //G4ThreeVector tlate(transform->NetTranslation());
   //cout <<  " tlate " << tlate << endl ;
   cout <<  "  g2l\n" << transform_rep(g2l) << endl ;
   cout <<  "  l2g\n" << transform_rep(l2g) << endl ;
   cout <<  "  gpos " << gpos << endl ;

   const G4ThreeVector pos(gpos) ;
   const G4ThreeVector dir(-gpos) ;
   const G4ThreeVector pol ;
   const float time = 1. ;
   const float wavelength = 550. ;

   transport->CollectPhoton( pos, dir, pol, time, wavelength, pmtid );
}


void DsChromaStackAction_NewStage()
{
   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   G4DAETransport* tra = chroma->GetTransport();

   tra->GetPhotons()->Print();
   tra->GetPhotons()->Details(0);

   chroma->Propagate(-1); // not >0  fakes the propagation, ie just passes all photons off as hits
}


void Mockup_EndEvent(G4HCofThisEvent* HCE )
{
   G4DAEChroma::GetG4DAEChroma()->GetSensDet()->EndOfEvent(HCE);     // G4 calls this
}



int main()
{
    Mockup_DetDesc_SD();

    string transport = "G4DAECHROMA_CLIENT_CONFIG" ;
    string cachekey = "DAE_NAME_DYB_TRANSFORMCACHE" ;
    string sensdet = "DsPmtSensDet" ;

    DsChromaRunAction_BeginOfRunAction( transport, cachekey, sensdet , "", "" );
    G4SDManager::GetSDMpointer()->ListTree();

    G4HCofThisEvent* HCE = Mockup_NewEvent();


    G4DAETransformCache* cache = G4DAEChroma::GetG4DAEChroma()->GetTransformCache();

    // mockup a hit for every PMT    
    for( size_t index = 0 ; index < cache->GetSize() ; ++index ) DsChromaStackAction_ClassifyNewTrack(cache->GetKey(index));

    //DsChromaStackAction_ClassifyNewTrack(cache->GetKey(0));
    //DsChromaStackAction_ClassifyNewTrack(0x1010101);


    DsChromaStackAction_NewStage();

    Mockup_EndEvent(HCE);
}




