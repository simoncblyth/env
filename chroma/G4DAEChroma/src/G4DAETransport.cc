#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"

#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"

#include "G4Track.hh"
#include "G4VProcess.hh"

#include <iostream>

using namespace std ; 

G4DAETransport::G4DAETransport(const char* envvar) :
    m_socket(NULL),
    m_photons(NULL),
    m_hits(NULL)
{ 
#ifdef WITH_CHROMA_ZMQ
   m_socket = new G4DAESocketBase(envvar) ;

   // TODO: make this NULL and move to always using SetPhotons ?
   m_photons = (Photons_t*)new G4DAEPhotonList(100) ;   

#endif
}

G4DAETransport::~G4DAETransport()
{
#ifdef WITH_CHROMA_ZMQ
   delete m_photons ; 
   delete m_hits  ; 
   delete m_socket ; 
#endif
}


Photons_t* G4DAETransport::GetPhotons(){ 
    return m_photons ; 
}
Photons_t* G4DAETransport::GetHits(){ 
    return m_hits ; 
}


void G4DAETransport::SetPhotons(Photons_t* photons)
{
   delete m_photons ; 
   m_photons = photons ; 
}
void G4DAETransport::SetHits(Photons_t* hits)
{
   delete m_hits ; 
   m_hits = hits ; 
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

#ifdef VERBOSE
   cout << "G4DAETransport::Propagate batch_id " << batch_id <<  " size " << size <<  endl ;   
   m_photons->Print();
#endif


  if( batch_id > 0 )
  { 
#ifdef VERBOSE
      cout << "G4DAETransport::Propagate : SendReceiveObject " <<  endl ;   
#endif
      m_hits = reinterpret_cast<Photons_t*>(m_socket->SendReceiveObject(m_photons));
  } 
  else 
  {
      cout << "G4DAETransport::Propagate : fake Send/Recv " << endl ; 
      m_hits = m_photons ;  // potential double delete, but just for debug 
  } 
  std::size_t nhits = m_hits->GetCount();
  return nhits ; 

}


