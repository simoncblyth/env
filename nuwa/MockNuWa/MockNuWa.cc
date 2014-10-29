#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
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

#ifndef NOT_NUWA
    ITouchableToDetectorElement* t2de = tool<ITouchableToDetectorElement>(m_t2deName);
    DybG4DAEGeometry* geometry  = new DybG4DAEGeometry(t2de, m_idParameter.c_str());
#else
    G4DAEGeometry* geometry  = new G4DAEGeometry();
#endif

    // hmm control archivedir via envvar  
#ifndef NOT_NUWA
    geometry->CreateTransformCache(NULL); 
    geometry->ArchiveCache("DybG4DAEGeometry.cache"); 
#else
    // hmm when have the transform cache no need to load the GDML
    geometry->LoadCache("/tmp/DybG4DAEGeometry.cache");  
#endif
    chroma->SetGeometry( geometry );  

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

    cout << "DsChromaRunAction::BeginOfRunAction AddNewDetector [" << sensdet->GetName() << "]" << endl ; 

    G4DAETransport* transport = G4DAETransport::MakeTransport(m_transport.c_str());
    chroma->SetTransport( transport );


    G4DAETransformCache* cache = geometry->GetCache();
    if( cache )
    {
        cache->Dump(); 
    }


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

   cout << "DsChromaStackAction_ClassifyNewTrack " << (void*)pmtid << endl ; 

   const G4ThreeVector pos ;
   const G4ThreeVector dir ;
   const G4ThreeVector pol ;
   const float time = 1. ;
   const float wavelength = 550. ;

   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
   G4DAETransport* tra = chroma->GetTransport();
  
   /*
   G4DAEGeometry*  geo = chroma->GetGeometry();
   G4AffineTransform* transform = geo->GetSensorTransform(pmtid);
   if( transform ) cout <<  " tlate " << transform->NetTranslation() << endl ;
   */

   tra->CollectPhoton( pos, dir, pol, time, wavelength, pmtid );
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


    G4DAETransformCache* cache = G4DAEChroma::GetG4DAEChroma()->GetGeometry()->GetCache();
    // mockup a hit for every PMT    
    for( size_t index = 0 ; index < cache->GetSize() ; ++index )
    {
        Key_t key = cache->GetKey(index);
        DsChromaStackAction_ClassifyNewTrack(key);
    }

    DsChromaStackAction_NewStage();

    Mockup_EndEvent(HCE);
}




