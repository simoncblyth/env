#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"

#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"

#include "G4Track.hh"
#include "G4VProcess.hh"

#include <iostream>


#define G4DAETRANSPORT_VERBOSE

using namespace std ; 

G4DAETransport::G4DAETransport(const char* envvar) :
    m_socket(NULL),
    m_photons(NULL),
    m_hits(NULL),
    m_cerenkov(NULL),
    m_scintillation(NULL),
    m_handshake(NULL)
{ 
#ifdef WITH_CHROMA_ZMQ
   m_socket = new G4DAESocketBase(envvar) ;

   // the arrays grow as needed

   m_photons = (G4DAEPhotons*)new G4DAEPhotonList(10000) ;   

   m_cerenkov = new G4DAECerenkovStepList(10000);

   m_scintillation = new G4DAEScintillationStepList(10000);


#endif
}

G4DAETransport::~G4DAETransport()
{
#ifdef WITH_CHROMA_ZMQ
   delete m_photons ; 
   delete m_hits  ; 
   delete m_socket ; 
   delete m_cerenkov ; 
   delete m_scintillation ; 
   delete m_handshake ; 
#endif
}


G4DAEMetadata* G4DAETransport::GetHandshake(){ 
    return m_handshake ; 
}

void G4DAETransport::Handshake(G4DAEMetadata* request)
{
    if(!request)
    {
        request = new G4DAEMetadata("{}"); 
    } 
    m_handshake = reinterpret_cast<G4DAEMetadata*>(m_socket->SendReceiveObject(request));
    if(m_handshake)
    {
        m_handshake->Print("G4DAETransport::Handshake");
    } 
    else
    {
        request->Print("G4DAETransport::Handshake FAILED with request:");
    }
}

G4DAEPhotons* G4DAETransport::GetPhotons(){ 
    return m_photons ; 
}
G4DAEPhotons* G4DAETransport::GetHits(){ 
    return m_hits ; 
}
G4DAECerenkovStepList* G4DAETransport::GetCerenkovStepList(){ 
    return m_cerenkov ; 
}
G4DAEScintillationStepList* G4DAETransport::GetScintillationStepList(){ 
    return m_scintillation ; 
}




void G4DAETransport::SetPhotons(G4DAEPhotons* photons)
{
   delete m_photons ; 
   m_photons = photons ; 
}
void G4DAETransport::SetHits(G4DAEPhotons* hits)
{
   delete m_hits ; 
   m_hits = hits ; 
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
#endif
}


void G4DAETransport::CollectPhoton(const G4Track* aPhoton )
{

#ifdef WITH_CHROMA_ZMQ
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4String pname="-";
   const G4VProcess* process = aPhoton->GetCreatorProcess();
   if(process) pname = process->GetProcessName();
   assert( pname == "Cerenkov" || pname == "Scintillation" );

   /* 
   G4cout << " OP : " 
          << " ProcessName " << pname 
          << " ParentID "    << aPhoton->GetParentID() 
          << " TrackID "     << aPhoton->GetTrackID() 
          << " KineticEnergy " << aPhoton->GetKineticEnergy() 
          << " TotalEnergy " << aPhoton->GetTotalEnergy() 
          << " TrackStatus " << aPhoton->GetTrackStatus() 
          << " CurrentStepNumber " << aPhoton->GetCurrentStepNumber() 
          << G4endl;
   */

   G4ThreeVector pos = aPhoton->GetPosition()/mm ;
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ;
   G4ThreeVector pol = aPhoton->GetPolarization() ;

   const float time = aPhoton->GetGlobalTime()/ns ;
   const float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   // weight
   // flags 
   // trackid ? 

   CollectPhoton( pos, dir, pol, time, wavelength );

#endif
}



// useful to retain this for easy standalone debug
void G4DAETransport::CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid)
{
    m_photons->AddPhoton(  pos, dir, pol, time, wavelength, pmtid );
}



std::size_t G4DAETransport::Propagate(int batch_id)
{
   size_t size = m_photons ? m_photons->GetCount() : 0 ;
   if(size == 0){
       cout << "G4DAETransport::Propagate SKIP no/empty photons list  " <<  endl;   
       return 0 ;
   }

#ifdef G4DAETRANSPORT_VERBOSE
   cout << "G4DAETransport::Propagate batch_id " << batch_id <<  " size " << size <<  endl ;   
   m_photons->Print();
#endif


  if( batch_id > 0 )
  { 
#ifdef G4DAETRANSPORT_VERBOSE
      cout << "G4DAETransport::Propagate : SendReceiveObject " <<  endl ;   
#endif
      m_hits = reinterpret_cast<G4DAEPhotons*>(m_socket->SendReceiveObject(m_photons));
  } 
  else 
  {
      cout << "G4DAETransport::Propagate : fake Send/Recv " << endl ; 
      m_hits = m_photons ;  // potential double delete, but just for debug 
  } 
  std::size_t nhits = m_hits ? m_hits->GetCount() : 0 ;
  return nhits ; 

}


