#include "DsChromaRunAction.h"

#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"

#include "G4DAEChroma/G4DAEChroma.hh"


// Needed for chroma initialization in the icc

#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAEMaterialMap.hh"

#include "DybG4DAEGeometry.h"
#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

#include "G4DataHelpers/ITouchableToDetectorElement.h"

#include <stdio.h>  
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

    declareProperty("cachekey",m_cachekey="G4DAECHROMA_CACHE_DIR",
                    "Name of envvar pointing to directory where cache is written to");

    declareProperty("databasekey",m_databasekey="G4DAECHROMA_DATABASE_PATH",
                    "Name of envvar pointing to config/monitoring sqlite3 database path");

    declareProperty("TouchableToDetelem", m_t2deName = "TH2DE",
                    "The ITouchableToDetectorElement to use to resolve sensor.");

    declareProperty("PackedIdPropertyName",m_idParameter="PmtID",
                    "The name of the user property holding the PMT ID.");

    declareProperty("EnableChroma",m_enableChroma = false, 
                    "Enable GPU optical photon propagation with Chroma, requires paired DsChromaStackAction");

    declareProperty("ChromaFlags",m_chromaFlags = "", 
                    "Delimited String to be parsed into bitfield controlling G4DAEChroma");


}

DsChromaRunAction::~DsChromaRunAction()
{
}

//
// ugly code inclusion allows common source for Chroma initialization 
// for both NuWa and MockNuWa running  
//
#include "DsChromaRunAction_BeginOfRunAction.icc"
void DsChromaRunAction::BeginOfRunAction( const G4Run* run )
{
    G4DAEChroma::GetG4DAEChroma()->BeginOfRun(run);

    ITouchableToDetectorElement* t2de = tool<ITouchableToDetectorElement>(m_t2deName);
    DsChromaRunAction_BeginOfRunAction( 
              m_transport, 
              m_cachekey, 
              m_sensdet, 
              m_databasekey, 
              t2de, 
              m_idParameter,
              m_enableChroma,
              m_chromaFlags );

}

void DsChromaRunAction::EndOfRunAction( const G4Run* run )
{
    G4DAEChroma::GetG4DAEChroma()->EndOfRun(run);
}


