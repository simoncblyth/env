#include "DsChromaRunAction.h"
#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"
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

};

DsChromaRunAction::~DsChromaRunAction()
{
};

void DsChromaRunAction::BeginOfRunAction( const G4Run* run )
{
    assert(run);

    // see env/nuwa/MockNuWa/MockNuWa.cc

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    chroma->BeginOfRun(run);
    chroma->Configure(m_transport.c_str(), m_sensdet.c_str(), m_geometry.c_str());

    DybG4DAECollector* col = new DybG4DAECollector ;
    G4DAESensDet* sd = chroma->GetSensDet();
    sd->SetCollector(col); 

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    cout << "DsChromaRunAction::BeginOfRunAction " << sd->GetName() << endl ; 

    // ?check if already added 
    SDMan->AddNewDetector( sd );


};

void DsChromaRunAction::EndOfRunAction( const G4Run* run )
{
    G4DAEChroma::GetG4DAEChroma()->EndOfRun(run);
    assert(run);
};


