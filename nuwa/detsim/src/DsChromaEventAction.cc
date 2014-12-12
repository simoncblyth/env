#include "DsChromaEventAction.h"

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"

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

void DsChromaEventAction::BeginOfEventAction( const G4Event* /*event*/ )
{
    m_t0 = G4DAEMetadata::RealTime();
    printf("DsChromaEventAction::BeginOfEventAction t0 %f \n", m_t0 );
    m_map["BeginOfEvent"] = G4DAEMetadata::TimeStampLocal();
}

void DsChromaEventAction::EndOfEventAction( const G4Event* /*event*/ )
{
    double te = G4DAEMetadata::RealTime();
    double td = te - m_t0 ;
    printf("DsChromaEventAction::EndOfEventAction te %f t0 %f td %f \n", te, m_t0, td );

    m_map["EndOfEvent"] = G4DAEMetadata::TimeStampLocal();
    m_map["DurationOfEvent"] = toStr<double>(td) ;  

    m_map["COLUMNS"] = "BeginOfEvent:s,EndOfEvent:s,DurationOfEvent:f" ;

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
    G4DAEDatabase* db = chroma->GetDatabase(); 
    G4DAEMetadata* meta = chroma->GetMetadata(); 

    meta->AddMap("eventaction",m_map);
    meta->Print("#chromameta");

   
    G4DAEMetadata* m = meta->GetLink();
    while(m)
    {  
        m->Print("#chromameta_linked") ;
        m = m->GetLink();
    }   



    if(db)
    {
        db->Insert(meta, "tevent",  "BeginOfEvent,EndOfEvent,DurationOfEvent" );
    }
    else
    {
        printf("DsChromaEventAction::EndOfEventAction db NULL \n");
    }
}


