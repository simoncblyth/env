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


#ifndef NOT_NUWA
void DsChromaRunAction_BeginOfRunAction( const string& m_transport, const string& m_geometry, const string& m_sensdet, const string& m_t2deName, const string& m_idParameter )
#else
void DsChromaRunAction_BeginOfRunAction( const string& m_transport, const string& m_geometry, const string& m_sensdet, const string& /*m_t2deName*/, const string& /*m_idParameter*/ )
#endif
{ 
   /////////// DsChromaRunAction::BeginOfRun //////////////////////////
   ///
   ///  * configure G4DAEChroma singleton
   ///  * add trojan SD, providing backdoor for adding "GPU" hits 
   ///
   /////////////////////////////////////////////////////////////////////

    cout << "DsChromaRunAction_BeginOfRunAction" << endl ;
    cout << "\t transport " << m_transport << endl ;
    cout << "\t geometry  " << m_geometry  << endl ;
    cout << "\t sensdet   " << m_sensdet << endl ;

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();


    ///
    ///  geometry needs re-architecting before adding 
    ///  G4DAE and IDMAP export functionality 
    ///

#ifndef NOT_NUWA
    ITouchableToDetectorElement* t2de = tool<ITouchableToDetectorElement>(m_t2deName);
    DybG4DAEGeometry* geometry  = new DybG4DAEGeometry(t2de, m_idParameter.c_str());
#else
    G4DAEGeometry* geometry  = new G4DAEGeometry();
#endif

    string m_cachekey = "DAE_NAME_DYB_TRANSFORMCACHE" ;
    const char* cachedir = getenv(m_cachekey.c_str());
    assert( cachedir ); // envvar must be provided pointing to transform cache directory 

#ifndef NOT_NUWA
    geometry->CreateTransformCache(NULL); 
    geometry->ArchiveCache(cachedir); 
#else
    // hmm when have the transform cache no need to load the GDML
    //geometry->LoadCache(cachedir);  

    if(G4DAETransformCache::Exists(cachedir))
    { 
        G4DAETransformCache* cache = G4DAETransformCache::Load(cachedir); 
        //cache->Dump(); 
        chroma->SetTransformCache(cache);
    }
    else
    {
         assert(0); // transform cache is required
    } 
#endif
    chroma->SetGeometry( geometry );  



     ///  sensdet 

    const char* target = m_sensdet.c_str() ; 
    string trojan = "trojan_" ;
    trojan += target ;

    G4DAESensDet*  sensdet = G4DAESensDet::MakeSensDet(trojan.c_str(), target );
    sensdet->Print(); 

    DybG4DAECollector* collector = new DybG4DAECollector ;
    sensdet->SetCollector(collector); 

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    SDMan->AddNewDetector( sensdet );
    chroma->SetSensDet( sensdet );  


    ////// transport 

    cout << "DsChromaRunAction::BeginOfRunAction AddNewDetector [" << sensdet->GetName() << "]" << endl ; 

    G4DAETransport* transport = G4DAETransport::MakeTransport(m_transport.c_str());
    chroma->SetTransport( transport );


}



void Mockup_DetDesc_SD()
{
   ////////// Mockup SD matching NuWa/GiGa/DetDesc ///////////

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
   /////////////  G4EventManager? ////////////////////////////

   G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 
   return HCE ; 
}


void DsChromaStackAction_ClassifyNewTrack(int pmtid)
{
   // mock OP know the PMT in their destiny 
   //cout << "DsChromaStackAction_ClassifyNewTrack " << (void*)pmtid << endl ; 

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   G4DAETransport*   transport = chroma->GetTransport();
   G4DAETransformCache*  cache = chroma->GetTransformCache();

   G4AffineTransform* pg2l = cache->GetSensorTransform(pmtid);
   assert(pg2l);

   // i think this manipulation is correct, but I have 
   // scrambled the transforms in the cache via wrong ordering for
   // row vector convention  

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
    string geometry = "" ; // "DAE_NAME_DYB_GDML" ;
    string sensdet = "DsPmtSensDet" ;

    DsChromaRunAction_BeginOfRunAction( transport, geometry, sensdet , "", "" );
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




