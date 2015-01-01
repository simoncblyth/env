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
    // hmm maybe add less manual timestamping, eg using map<string,double> timestamp structure 

    double te ; 
    {   
        te = G4DAEMetadata::RealTime();
        double d = te - m_t0 ;
        printf("DsChromaEventAction::EndOfEventAction-Head te %f t0 %f te-t0 %f \n", te, m_t0, d );
        m_map["EndOfEvent"] = G4DAEMetadata::TimeStampLocal();
        m_map["DurationOfEvent"] = toStr<double>(d) ;  
        m_map["COLUMNS"] = "BeginOfEvent:s,EndOfEvent:s,DurationOfEvent:f,TailOfEvent:f,DurationOfTail:f" ;
    }  

    ChromaProcessing();

    double tf ; 
    {   
        tf = G4DAEMetadata::RealTime();
        double d = tf - te ;
        printf("DsChromaEventAction::EndOfEventAction-Tail te %f tf %f tf-te %f \n", tf, te, d );
        m_map["TailOfEvent"] = G4DAEMetadata::TimeStampLocal();
        m_map["DurationOfTail"] = toStr<double>(d) ;  
    }  


    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
    G4DAEMetadata* meta = chroma->GetMetadata(); 
    {
        meta->AddMap("eventaction",m_map);
        meta->Print("#chromameta");
        meta->PrintLinks("#chromameta_links");
    }   

    {
        G4DAEDatabase* db = chroma->GetDatabase(); 
        if(db)
        {
            db->Insert(meta, "tevent",  "BeginOfEvent,EndOfEvent,DurationOfEvent" );
        }
        else
        {
            printf("DsChromaEventAction::EndOfEventAction db NULL \n");
        }
    }
}


void DsChromaEventAction::ChromaProcessing()
{

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
    if(chroma->HasFlag(G4DAEChroma::FLAG_G4CERENKOV_COLLECT_STEP))
    {
        printf("FLAG_G4CERENKOV_COLLECT_STEP\n"); 
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
        printf("FLAG_G4SCINTILLATION_COLLECT_STEP\n"); 
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
        printf("FLAG_G4CERENKOV_COLLECT_PHOTON\n"); 
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
        printf("FLAG_G4SCINTILLATION_COLLECT_PHOTON\n"); 
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

}




