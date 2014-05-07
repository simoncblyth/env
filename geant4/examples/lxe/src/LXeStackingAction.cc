//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
#include "LXeStackingAction.hh"
#include "LXeUserEventInformation.hh"
#include "LXeSteppingAction.hh"

#include "G4ios.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4Track.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"

#ifdef WITH_CHROMA_ZMQ
#include "ChromaPhotonList.hh"
#include "ZMQRoot.hh"
#endif


LXeStackingAction::LXeStackingAction() :  fZMQRoot(NULL), fPhotonList(NULL), fPhotonList2(NULL)
{
  G4cout << "LXeStackingAction::LXeStackingAction " <<  G4endl;   

#ifdef WITH_CHROMA_ZMQ
  fZMQRoot = new ZMQRoot("LXE_CLIENT_CONFIG") ; 
  fPhotonList = new ChromaPhotonList ;   
#endif
}

LXeStackingAction::~LXeStackingAction()
{
  G4cout << "LXeStackingAction::~LXeStackingAction " <<  G4endl;   

#ifdef WITH_CHROMA_ZMQ
  if(fPhotonList2) delete fPhotonList2 ;  
  delete fPhotonList ;  
  delete fZMQRoot ; 
#endif
}


void LXeStackingAction::CollectPhoton(const G4Track* aPhoton )
{
#ifdef WITH_CHROMA_ZMQ
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ; 
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ; 
   G4ThreeVector pol = aPhoton->GetPolarization() ;
   float time = aPhoton->GetGlobalTime()/ns ;
   float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   fPhotonList->AddPhoton( pos, dir, pol, time, wavelength );
   fPhotonList->Print();
#endif
}

G4ClassificationOfNewTrack LXeStackingAction::ClassifyNewTrack(const G4Track * aTrack){
 
  G4cout << "LXeStackingAction::ClassifyNewTrack TrackID " << aTrack->GetTrackID() << " ParentID " << aTrack->GetParentID() <<  G4endl;   

  G4bool is_op = aTrack->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition() ; 
  G4bool is_secondary = aTrack->GetParentID()>0 ; 

  G4EventManager* evtmgr = G4EventManager::GetEventManager();
  const G4Event* event = evtmgr->GetConstCurrentEvent() ; 
  LXeUserEventInformation* eventInformation = (LXeUserEventInformation*)event->GetUserInformation();
  
  if(is_op){ 

      CollectPhoton( aTrack );

      if(is_secondary){
         G4String procname = aTrack->GetCreatorProcess()->GetProcessName() ;
         G4cout << "LXeStackingAction::ClassifyNewTrack OP Secondary from " << procname << G4endl;  

         //Count what process generated the optical photons
         if(procname=="Scintillation") eventInformation->IncPhotonCount_Scint();
         else if(procname=="Cerenkov") eventInformation->IncPhotonCount_Ceren();
      }
  }
  return fUrgent;
}

void LXeStackingAction::NewStage(){

  G4cout << "LXeStackingAction::NewStage" << G4endl;   

#ifdef WITH_CHROMA_ZMQ

  G4cout << "::NewStage SendObject " <<  G4endl;   
  fZMQRoot->SendObject(fPhotonList);

  G4cout << "::NewStage ReceiveObject, waiting... " <<  G4endl;   
  fPhotonList2 = (ChromaPhotonList*)fZMQRoot->ReceiveObject();

  fPhotonList->Details();
  fPhotonList2->Details();
#endif

}

void LXeStackingAction::PrepareNewEvent(){ 

  G4cout << "LXeStackingAction::PrepareNewEvent" << G4endl;   

}








