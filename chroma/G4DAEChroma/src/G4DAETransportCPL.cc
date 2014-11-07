#include "G4DAEChroma/G4DAETransportCPL.hh"

#include "G4Track.hh"
#include "G4VProcess.hh"

#include <iostream>

using namespace std ; 

#ifdef WITH_CHROMA_ZMQ
#include "Chroma/ChromaPhotonList.hh"
#include "ZMQRoot/ZMQRoot.hh"
#endif


G4DAETransportCPL::G4DAETransportCPL(const char* envvar) :
    fZMQRoot(0),
    fPhotonList(0),
    fPhotonHits(0)
{ 
#ifdef WITH_CHROMA_ZMQ
  fPhotonList = new ChromaPhotonList;   
  fZMQRoot = new ZMQRoot(envvar);  
#endif
}

G4DAETransportCPL::~G4DAETransportCPL()
{
#ifdef WITH_CHROMA_ZMQ
   if(fPhotonList)  delete fPhotonList ; 
   if(fPhotonHits)  delete fPhotonHits ; 
   if(fZMQRoot)     delete fZMQRoot ; 
#endif
}


ChromaPhotonList* G4DAETransportCPL::GetPhotons(){ 
    return fPhotonList ; 
}
ChromaPhotonList* G4DAETransportCPL::GetHits(){ 
    return fPhotonHits ; 
}


void G4DAETransportCPL::ClearAll()
{
#ifdef WITH_CHROMA_ZMQ
 if(fPhotonList){ 
      cout<< "::ClearAll fPhotonList  "<<endl;
      fPhotonList->ClearAll(); 
  }
  if(fPhotonHits){ 
      cout<< "::ClearAll fPhotonHits "<<endl;
      fPhotonHits->ClearAll(); 
  } 
#endif
}


void G4DAETransportCPL::CollectPhoton(const G4Track* aPhoton )
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

   CollectPhoton( pos, dir, pol, time, wavelength );

#endif
}


void G4DAETransportCPL::CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid)
{
   fPhotonList->AddPhoton( 
              pos.x(), pos.y(), pos.z(),
              dir.x(), dir.y(), dir.z(),
              pol.x(), pol.y(), pol.z(), 
              time, 
              wavelength, 
              pmtid );
}




std::size_t G4DAETransportCPL::Propagate(int batch_id)
{
  fPhotonList->SetUniqueID(batch_id);

  cout << "::Propagate fPhotonList " << batch_id <<  endl ;   
  fPhotonList->Print(); 
  std::size_t size = fPhotonList->GetSize(); 
  if(size == 0){
      cout << "::Propagate Skip send/recv for empty CPL " <<  endl;   
      return 0 ;
  }

  if( batch_id > 0 )
  { 
      cout << "G4DAETransportCPL::Propagate : SendObject " <<  endl ;   
      fZMQRoot->SendObject(fPhotonList);

      cout << "G4DAETransportCPL::Propagate : ReceiveObject, waiting... " <<  endl;   
      fPhotonHits = (ChromaPhotonList*)fZMQRoot->ReceiveObject();
  } 
  else 
  {
      cout << "G4DAETransportCPL::Propagate : fake Send/Recv " << endl ; 
      fPhotonHits = fPhotonList ; 
  } 
  std::size_t nhits = fPhotonHits->GetSize();
  return nhits ; 
}



