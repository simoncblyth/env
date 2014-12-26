#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"

#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEScintillationPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovPhotonList.hh"

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include <string>
#include <iostream>
#include <stdio.h>

#define G4DAETRANSPORT_VERBOSE

using namespace std ; 

G4DAETransport::G4DAETransport(const char* envvar) :
    m_socket(NULL),
    m_handshake(NULL),
    m_cerenkov(NULL),
    m_scintillation(NULL),
    m_photons(NULL),
    m_scintillation_photons(NULL),
    m_cerenkov_photons(NULL),
    m_hits(NULL),
    m_verbosity(3)
{ 
#ifdef WITH_CHROMA_ZMQ
   m_socket = new G4DAESocketBase(envvar) ;
#endif
}

G4DAETransport::~G4DAETransport()
{
#ifdef WITH_CHROMA_ZMQ
   delete m_hits  ; 
   delete m_cerenkov_photons ; 
   delete m_scintillation_photons ; 
   delete m_photons ; 
   delete m_scintillation ; 
   delete m_cerenkov ; 
   delete m_handshake ; 
   delete m_socket ; 
#endif
}


int G4DAETransport::GetVerbosity()
{
    return m_verbosity ;
}

void G4DAETransport::SetVerbosity(int verbosity)
{
     m_verbosity = verbosity ;
}



G4DAEMetadata* G4DAETransport::GetHandshake()
{ 
    return m_handshake ; 
}

void G4DAETransport::Handshake(G4DAEMetadata* request)
{
    if(!request) request = new G4DAEMetadata("{}"); 

    if( m_verbosity > 0 )
    request->Print("G4DAETransport::Handshake waiting for handshake response:");

    m_handshake = reinterpret_cast<G4DAEMetadata*>(m_socket->SendReceiveObject(request));

    if(!m_handshake)
    {
        request->Print("G4DAETransport::Handshake FAILED with request:");
        return ;  
    }

    //m_handshake->PrintMap("G4DAETransport::Handshake PrintMap");
}






G4DAECerenkovStepList* G4DAETransport::GetCerenkovStepList()
{ 
    if(!m_cerenkov) m_cerenkov = new G4DAECerenkovStepList(10000);
    return m_cerenkov ; 
}
G4DAEScintillationStepList* G4DAETransport::GetScintillationStepList()
{ 
    if(!m_scintillation) m_scintillation = new G4DAEScintillationStepList(10000);
    return m_scintillation ; 
}


G4DAEPhotonList* G4DAETransport::GetHits()
{ 
    // hits only created from socket response
    return m_hits ; 
}

G4DAEPhotonList* G4DAETransport::GetPhotons()
{ 
    if(!m_photons) m_photons = new G4DAEPhotonList(10000);
    return m_photons ; 
}
G4DAEScintillationPhotonList* G4DAETransport::GetScintillationPhotonList()
{ 
    if(!m_scintillation_photons) m_scintillation_photons = new G4DAEScintillationPhotonList(10000);
    return m_scintillation_photons ; 
}
G4DAECerenkovPhotonList* G4DAETransport::GetCerenkovPhotonList()
{ 
    if(!m_cerenkov_photons) m_cerenkov_photons = new G4DAECerenkovPhotonList(10000);
    return m_cerenkov_photons ; 
}



void G4DAETransport::SetHits(G4DAEPhotonList* hits)
{ 
    delete m_hits ; 
    m_hits = hits ; 
}
void G4DAETransport::SetPhotons(G4DAEPhotonList* photons)
{
    delete m_photons ; 
    m_photons = photons ; 
}
void G4DAETransport::SetScintillationPhotonList(G4DAEScintillationPhotonList* scintillation_photons)
{
    delete m_scintillation_photons ; 
    m_scintillation_photons = scintillation_photons ; 
}
void G4DAETransport::SetCerenkovPhotonList(G4DAECerenkovPhotonList* cerenkov_photons)
{
    delete m_cerenkov_photons ; 
    m_cerenkov_photons = cerenkov_photons ; 
}


void G4DAETransport::SetCerenkovStepList(G4DAECerenkovStepList* cerenkov)
{
   delete m_cerenkov ; 
   m_cerenkov = cerenkov ; 
}
void G4DAETransport::SetScintillationStepList(G4DAEScintillationStepList* scintillation)
{
   delete m_scintillation ; 
   m_scintillation = scintillation ; 
}



void G4DAETransport::ClearAll()
{
#ifdef WITH_CHROMA_ZMQ
    if(m_photons)
    {
        m_photons->ClearAll();   
    }
    if(m_hits)
    {
        m_hits->ClearAll();   
    }
    if(m_cerenkov)
    {
        m_cerenkov->ClearAll();   
    }
    if(m_scintillation)
    {
        m_scintillation->ClearAll();   
    }
    if(m_scintillation_photons)
    {
        m_scintillation_photons->ClearAll();   
    }
    if(m_cerenkov_photons)
    {
        m_cerenkov_photons->ClearAll();   
    }
#endif
}




std::size_t G4DAETransport::ProcessCerenkovSteps(int batch_id)
{
    return Process(batch_id, m_cerenkov );
}
std::size_t G4DAETransport::ProcessScintillationSteps(int batch_id)
{
    return Process(batch_id, m_scintillation );
}
std::size_t G4DAETransport::ProcessCerenkovPhotons(int batch_id)
{
    return Process(batch_id, m_cerenkov_photons );
}
std::size_t G4DAETransport::ProcessScintillationPhotons(int batch_id)
{
    return Process(batch_id, m_scintillation_photons );
}





std::size_t G4DAETransport::Propagate(int batch_id)
{
    return Process(batch_id, m_photons );
}



G4DAEArrayHolder* G4DAETransport::ProcessRaw(int /*batch_id*/, G4DAEArrayHolder* request)
{
    size_t size = request ? request->GetCount() : 0 ;
    if(size == 0){
        request->Print("G4DAETransport::ProcessRaw EMPTY request");
        return NULL ;
    }

    if(m_verbosity > 0){
        request->Print("G4DAETransport::ProcessRaw request");
    } 

    G4DAEArrayHolder* response = m_socket->SendReceive(request);
    return response ; 
}



std::size_t G4DAETransport::Process(int batch_id, G4DAEArrayHolder* request)
{
    G4DAEArrayHolder* response = ProcessRaw(batch_id, request );
    G4DAEPhotonList* hits = NULL ; 

    if(response)
    {
        if(m_verbosity > 0 ) response->Print("G4DAETransport::Process response");
        hits = new G4DAEPhotonList(response);
    }
    else
    {
         cout << "G4DAETransport::Process response NULL " << endl ;  
    } 

    SetHits(hits);
    std::size_t count = hits ? hits->GetCount() : 0 ;
    return count ;

}









