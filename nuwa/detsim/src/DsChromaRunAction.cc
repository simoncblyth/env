#include "DsChromaRunAction.h"
#include "G4DAEChroma/G4DAEChroma.hh"

#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"
#include <stdlib.h>  
#include <assert.h>

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
    G4DAEChroma::GetG4DAEChroma()->BeginOfRun(run);
    assert(run);
    G4DAEChroma::GetG4DAEChroma()->Configure(m_transport.c_str(), m_sensdet.c_str(), m_geometry.c_str());
};

void DsChromaRunAction::EndOfRunAction( const G4Run* run )
{
    G4DAEChroma::GetG4DAEChroma()->EndOfRun(run);
    assert(run);
};


