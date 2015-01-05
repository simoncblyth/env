#include "DsChromaEventAction.h"

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEPmtHitList.hh"
#include "DybG4DAECollector.h"

#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"

#include <stdio.h>  
#include <stdlib.h>  
#include <assert.h>
#include <iostream>

DECLARE_TOOL_FACTORY( DsChromaEventAction );

DsChromaEventAction::DsChromaEventAction
( const std::string& type   ,
  const std::string& name   ,
  const IInterface*  parent ) 
  : GiGaEventActionBase( type , name , parent )
{  
}

DsChromaEventAction::~DsChromaEventAction()
{
}

void DsChromaEventAction::BeginOfEventAction( const G4Event* event )
{
    G4DAEChroma::GetG4DAEChroma()->BeginOfEvent(event);
}

void DsChromaEventAction::EndOfEventAction( const G4Event* event )
{
    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

    DybG4DAECollector* collector = (DybG4DAECollector*)chroma->GetCollector();
    collector->HarvestPmtHits();  // from G4 HC into PmtHitList

    G4DAEPmtHitList* phl = chroma->GetPmtHitList();
    phl->Save("1", "HIT", "DAE_G4PMTHIT_TEMPLATE_PATH");

    chroma->EndOfEvent(event);
}

