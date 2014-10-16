#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"

#ifdef WITH_CHROMA_ZMQ
#include "Chroma/ChromaPhotonList.hh"
#include "ZMQRoot/ZMQRoot.hh"
#endif

#include "G4VProcess.hh"
#include "G4AffineTransform.hh"
#include "G4TransportationManager.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"


G4DAEChroma* G4DAEChroma::fG4DAEChroma = 0;

G4DAEChroma* G4DAEChroma::GetG4DAEChroma()
{
  if(!fG4DAEChroma)
  {
     fG4DAEChroma = new G4DAEChroma;
  }
  return fG4DAEChroma;
}

G4DAEChroma* G4DAEChroma::GetG4DAEChromaIfExists()
{ 
  return fG4DAEChroma ;
}




G4DAEChroma::G4DAEChroma() :
    fZMQRoot(0),
    fPhotonList(0),
    fPhotonList2(0),
    fGeometry(0)
{ 
#ifdef WITH_CHROMA_ZMQ
  fPhotonList = new ChromaPhotonList;   
  fZMQRoot = new ZMQRoot("CSA_CLIENT_CONFIG");  //TODO: pass along this config from upper ctor ? change default G4DAECHROMA_CLIENT_CONFIG
#endif
}

G4DAEChroma::~G4DAEChroma()
{
#ifdef WITH_CHROMA_ZMQ
   if(fPhotonList)  delete fPhotonList ; 
   if(fPhotonList2) delete fPhotonList2 ; 
   if(fZMQRoot)     delete fZMQRoot ; 
#endif
}


void G4DAEChroma::SetGeometry(G4DAEGeometry* geo){
   fGeometry = geo ; 
}

G4DAEGeometry* G4DAEChroma::GetGeometry(){
   return fGeometry ;
}


void G4DAEChroma::ClearAll()
{
#ifdef WITH_CHROMA_ZMQ
 if(fPhotonList){ 
      G4cout<< "::ClearAll fPhotonList  "<<G4endl;
      fPhotonList->ClearAll(); 
  }
  if(fPhotonList2){ 
      G4cout<< "::ClearAll fPhotonList2 "<<G4endl;
      fPhotonList2->ClearAll(); 
  } 
#endif
}

void G4DAEChroma::CollectPhoton(const G4Track* aPhoton )
{
#ifdef WITH_CHROMA_ZMQ
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4String pname="-";
   const G4VProcess* process = aPhoton->GetCreatorProcess();
   if(process) pname = process->GetProcessName();
   G4cout << " OP : " 
          << " ProcessName " << pname 
          << " ParentID "    << aPhoton->GetParentID() 
          << " TrackID "     << aPhoton->GetTrackID() 
          << " KineticEnergy " << aPhoton->GetKineticEnergy() 
          << " TotalEnergy " << aPhoton->GetTotalEnergy() 
          << " TrackStatus " << aPhoton->GetTrackStatus() 
          << " CurrentStepNumber " << aPhoton->GetCurrentStepNumber() 
          << G4endl;

   assert( pname == "Cerenkov" || pname == "Scintillation" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ;
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ;
   G4ThreeVector pol = aPhoton->GetPolarization() ;
   float time = aPhoton->GetGlobalTime()/ns ;
   float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   fPhotonList->AddPhoton( 
              pos.x(), pos.y(), pos.z(),
              dir.x(), dir.y(), dir.z(),
              pol.x(), pol.y(), pol.z(), 
              time, 
              wavelength );

#endif
}

void G4DAEChroma::Propagate(G4int batch_id)
{
#ifdef WITH_CHROMA_ZMQ


  fPhotonList->SetUniqueID(batch_id);
  G4cout << "::Propagate fPhotonList " <<  G4endl ;   
  fPhotonList->Print(); 
  std::size_t size = fPhotonList->GetSize(); 

  if(size > 0)
  {
      G4cout << "::SendObject " <<  G4endl ;   
      fZMQRoot->SendObject(fPhotonList);
      G4cout << "::ReceiveObject, waiting... " <<  G4endl;   
      fPhotonList2 = (ChromaPhotonList*)fZMQRoot->ReceiveObject();
      G4cout << "::fPhotonList2 " <<  G4endl ;   
      fPhotonList2->Print();
      
      for( std::size_t index = 0 ; index < size ; index++ )
      {
          ProcessHit( fPhotonList2,  index );
      }   

  } 
  else 
  { 
      G4cout << "::Propagate Skip send/recv for empty CPL " <<  G4endl;   
  }
#else
      G4cout << "::Propagate : NEED TO RECOMPILE USING : -DWITH_CHROMA_ZMQ  " <<  G4endl;   
#endif
}

bool G4DAEChroma::ProcessHit( const ChromaPhotonList* cpl, std::size_t index )
{
#ifdef WITH_CHROMA_ZMQ
    Hit hit ; 

    cpl->GetPhoton( index, hit.gpos, hit.gdir, hit.gpol, hit.t, hit.wavelength, hit.pmtid );    

    hit.hitindex = index ;
    hit.volumeindex = 0 ; //dummy

    G4AffineTransform identity ;
    G4AffineTransform& trans = fGeometry ? fGeometry->GetNodeTransform(hit.volumeindex) : identity ;
    hit.LocalTransform(trans);

    hit.Print();
   
#else
    printf("need to recompile -DWITH_CHROMA_ZMQ \n");
#endif
    return true;
}



