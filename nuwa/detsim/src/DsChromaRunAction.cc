#include "DsChromaRunAction.h"

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransport.hh"

#include "DybG4DAEGeometry.h"
#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"


#include "G4DataHelpers/ITouchableToDetectorElement.h"

#include <stdlib.h>  
#include <assert.h>
#include <iostream>

using namespace std ; 

DECLARE_TOOL_FACTORY( DsChromaRunAction );

DsChromaRunAction::DsChromaRunAction
( const std::string& type   ,
  const std::string& name   ,
  const IInterface*  parent ) 
  : GiGaRunActionBase( type , name , parent )
{  
    declareProperty("transport",m_transport="G4DAECHROMA_CLIENT_CONFIG",
                    "Name of envvar holding the transport config string");

    declareProperty("sensdet",m_sensdet="DsPmtSensDet",
                    "Name of target SD where Chroma derived hits will be collected.");

    declareProperty("geometry",m_geometry="MEMORY",
                    "Used to determine where to load geometry from, default of MEMORY loads from in-memory volume heirarchy");

    declareProperty("TouchableToDetelem", m_t2deName = "TH2DE",
                    "The ITouchableToDetectorElement to use to resolve sensor.");

    declareProperty("PackedIdPropertyName",m_idParameter="PmtID",
                    "The name of the user property holding the PMT ID.");


};

DsChromaRunAction::~DsChromaRunAction()
{
};

void DsChromaRunAction::BeginOfRunAction( const G4Run* run )
{
    assert(run);

    /*
    Configuring external OP propagation and hit formation 
         
    see env/nuwa/MockNuWa/MockNuWa.cc

    */

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

    ITouchableToDetectorElement* t2de = tool<ITouchableToDetectorElement>(m_t2deName);
    DybG4DAEGeometry* geometry  = new DybG4DAEGeometry(t2de, m_idParameter.c_str());
    geometry->CreateTransformCache(NULL); 
    geometry->ArchiveCache("DybG4DAEGeometry.cache");  // hmm control this via envvar  


    const char* target = m_sensdet.c_str() ; 
    string trojan = "trojan_" ;
    trojan += target ;
    G4DAESensDet*  sensdet = G4DAESensDet::MakeSensDet(trojan.c_str(), target );
    DybG4DAECollector* collector = new DybG4DAECollector ;
    sensdet->SetCollector(collector); 

    G4DAETransport* transport = G4DAETransport::MakeTransport(m_transport.c_str());

    chroma->SetGeometry( geometry );  
    chroma->SetSensDet( sensdet );  
    chroma->SetTransport( transport );

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    cout << "DsChromaRunAction::BeginOfRunAction " << sensdet->GetName() << endl ; 

    SDMan->AddNewDetector( sensdet );
    chroma->BeginOfRun(run);

};

void DsChromaRunAction::EndOfRunAction( const G4Run* run )
{
    G4DAEChroma::GetG4DAEChroma()->EndOfRun(run);
    assert(run);
};


