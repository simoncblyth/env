#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAETrojanSensDet.hh"

#ifdef WITH_CHROMA_ZMQ
#include "Chroma/ChromaPhotonList.hh"
#include "ZMQRoot/ZMQRoot.hh"
#endif

#include "G4VProcess.hh"
#include "G4AffineTransform.hh"
#include "G4TransportationManager.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"

#include "G4SDManager.hh"

using namespace std ; 

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


G4DAEChroma::G4DAEChroma(const char* envvar) :
    fZMQRoot(0),
    fPhotonList(0),
    fPhotonList2(0),
    fGeometry(0)
{ 
#ifdef WITH_CHROMA_ZMQ
  fPhotonList = new ChromaPhotonList;   
  fZMQRoot = new ZMQRoot(envvar);  
  //TODO: pass along this config from upper ctor ? change default G4DAECHROMA_CLIENT_CONFIG
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

//
// create parasitic SD for adding hits to hitcollections of target SD, eg DsPmtSensDet
// need to register the Trojan SD at initialization time
// to gain access to HCE via Initialize 
// target parameter must match the name of an existing SD 
//

void G4DAEChroma::RegisterTrojanSD(const std::string& target)
{
    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    string name = "Trojan_" + target ;

    G4VSensitiveDetector* tsd = SDMan->FindSensitiveDetector(name, true);
    if( !tsd ){
        cout << "G4DAEChroma::RegisterTrojanSD AddNewDetector " << name << endl ;

        G4DAETrojanSensDet* tsd = new G4DAETrojanSensDet(name, target);
        tsd->SetGeometry(fGeometry);
        SDMan->AddNewDetector(tsd);
        SDMan->ListTree();
    } else {
       cout << "G4DAEChroma::RegisterTrojanSD SD named " << name << " exists already " << endl ;
    }
}

G4DAETrojanSensDet* G4DAEChroma::GetTrojanSD(const std::string& target)
{
    string name = "Trojan_" + target ;
    return (G4DAETrojanSensDet*)G4SDManager::GetSDMpointer()->FindSensitiveDetector(name, true);
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
   // defer this detail into CPL ?

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

void G4DAEChroma::Propagate(G4int batch_id, const std::string& target)
{
#ifdef WITH_CHROMA_ZMQ
  G4DAETrojanSensDet* TSD = GetTrojanSD(target);
  fPhotonList->SetUniqueID(batch_id);
  G4cout << "::Propagate fPhotonList " <<  G4endl ;   
  fPhotonList->Print(); 
  std::size_t size = fPhotonList->GetSize(); 

  if(size > 0)
  {
      cout << "::SendObject " <<  endl ;   
      fZMQRoot->SendObject(fPhotonList);
      cout << "::ReceiveObject, waiting... " <<  endl;   
      fPhotonList2 = (ChromaPhotonList*)fZMQRoot->ReceiveObject();
      TSD->CollectHits( fPhotonList2 );
  } 
  else 
  { 
      cout << "::Propagate Skip send/recv for empty CPL " <<  endl;   
  }
#else
      cout << "::Propagate : NEED TO RECOMPILE USING : -DWITH_CHROMA_ZMQ  " <<  endl;   
#endif
}





