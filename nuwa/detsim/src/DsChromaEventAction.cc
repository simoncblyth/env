#include "DsChromaEventAction.h"

#include "G4DAEChroma/G4DAEChroma.hh"

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
    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    chroma->BeginOfEvent(event);
}

void DsChromaEventAction::EndOfEventAction( const G4Event* event )
{
    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    chroma->EndOfEvent(event);
}

