#include "DsChromaEventAction.h"

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"

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


    // For debug only : Save GenStep Lists to file 

    G4DAECerenkovStepList* csl = chroma->GetCerenkovStepList(); 
    csl->Save("1");
    
    G4DAEScintillationStepList* ssl = chroma->GetScintillationStepList(); 
    ssl->Save("1");


    // For debug only : save Geant4 generated Scintillation and Cerenkov photons to NPY files

    G4DAEScintillationPhotonList* spl = chroma->GetScintillationPhotonList();   
    spl->Save("1");

    G4DAECerenkovPhotonList* cpl = chroma->GetCerenkovPhotonList();  
    cpl->Save("1");




    //  hmm the below get the GPU generated 

    size_t ncs = chroma->ProcessCerenkovSteps(1);    
    printf("ProcessCerenkovSteps ncs %zu \n", ncs); 
    G4DAEPhotonList* csp = chroma->GetHits();    // hmm GetResponse would be better
    csp->Print("response from ProcessCerenkovSteps ");
    csp->Save("1cs");


    size_t nss = chroma->ProcessScintillationSteps(1);    
    printf("ProcessScintillationSteps nss %zu \n", nss); 
    G4DAEPhotonList* css = chroma->GetHits();    // hmm GetResponse would be better
    css->Print("response from ProcessScintillationSteps ");
    css->Save("1ss");




 
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


