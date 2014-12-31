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


    if(chroma->HasFlag(G4DAEChroma::FLAG_G4CERENKOV_COLLECT_STEP))
    {
        G4DAECerenkovStepList* l = chroma->GetCerenkovStepList(); 
        l->SetKV("ctrl", "type", "cerenkov" );
        l->SetKV("ctrl", "evt", "1" );
        l->SetKV("ctrl", "threads_per_block", 512 );
        l->SetKV("ctrl", "noreturn", 1 );
        l->SetKV("ctrl", "sidesave", 1 );   // remote save of GPU generated photons

        size_t n = chroma->ProcessCerenkovSteps(1);    
        printf("ProcessCerenkovSteps FLAG_G4CERENKOV_COLLECT_STEP  n %zu \n", n); 
        G4DAEPhotonList* hits = chroma->GetHits();   
        if(hits)
        {
            hits->Print("response from ProcessCerenkovSteps ");
        }
        else
        { 
            printf("ProcessCerenkovSteps n %zu NULL hits  \n", n); 
        }
    }
    else
    {
        printf("ProcessCerenkovSteps FLAG_G4CERENKOV_COLLECT_STEP : SKIPPING  \n"); 
    }


    if(chroma->HasFlag(G4DAEChroma::FLAG_G4SCINTILLATION_COLLECT_STEP))
    {
        G4DAEScintillationStepList* l = chroma->GetScintillationStepList(); 
        l->SetKV("ctrl", "type", "scintillation" );
        l->SetKV("ctrl", "evt", "1" );
        l->SetKV("ctrl", "threads_per_block", 512 );
        l->SetKV("ctrl", "noreturn", 1 );
        l->SetKV("ctrl", "sidesave", 1 );   // remote save of GPU generated photons

        size_t n = chroma->ProcessScintillationSteps(1);    
        printf("ProcessScintillationSteps FLAG_G4SCINTILLATION_COLLECT_STEP  n %zu \n", n); 
        G4DAEPhotonList* hits = chroma->GetHits();   
        if(hits)
        {
            hits->Print("response from ProcessScintillationSteps ");
        }
        else
        {
            printf("ProcessScintillationSteps n %zu NULL hits \n", n); 
        } 
    }
    else
    {
        printf("ProcessScintillationSteps FLAG_G4SCINTILLATION_COLLECT_STEP : SKIPPING  \n"); 
    }




    if(chroma->HasFlag(G4DAEChroma::FLAG_G4CERENKOV_COLLECT_PHOTON))
    {
        G4DAECerenkovPhotonList* l = chroma->GetCerenkovPhotonList();  
        l->SetKV("ctrl", "type", "gopcerenkov" );  
        l->SetKV("ctrl", "evt", "1" );
        l->SetKV("ctrl", "onlycopy", 1 );  // just copy G4 generated photons to otherside
        //l->Save("1");


        size_t n = chroma->ProcessCerenkovPhotons(1);    
        printf("ProcessCerenkovPhotons FLAG_G4CERENKOV_COLLECT_PHOTON   n %zu \n", n); 
        G4DAEPhotonList* hits = chroma->GetHits();   
        if(hits)
        {
            hits->Print("response from ProcessCerenkovPhotons ");
        }
        else
        {
            printf("NULL hits from ProcessCerenkovPhotons n %zu \n", n); 
        }
    }
    else
    {
        printf("ProcessCerenkovPhotons FLAG_G4CERENKOV_COLLECT_PHOTON : SKIPPING  \n"); 
    }



    if(chroma->HasFlag(G4DAEChroma::FLAG_G4SCINTILLATION_COLLECT_PHOTON))
    {
        G4DAEScintillationPhotonList* l = chroma->GetScintillationPhotonList();   
        l->SetKV("ctrl", "type", "gopscintillation" );
        l->SetKV("ctrl", "evt", "1" );
        l->SetKV("ctrl", "onlycopy", 1 );  // just copy G4 generated photons to otherside
        //l->Save("1");

        size_t n = chroma->ProcessScintillationPhotons(1);    
        printf("ProcessScintillationPhotons FLAG_G4SCINTILLATION_COLLECT_PHOTON  n %zu \n", n); 
        G4DAEPhotonList* hits = chroma->GetHits();   
        if(hits)
        {
            hits->Print("response from ProcessScintillationPhotons ");
        }
        else
        {
            printf("NULL hits from ProcessScintillationPhotons n %zu \n", n); 
        } 
    }
    else
    {
        printf("ProcessScintillationPhotons FLAG_G4SCINTILLATION_COLLECT_PHOTON : SKIPPING  \n"); 
    }



 
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


