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

#include "ChromaPhotonList.hh"


LXeStackingAction::LXeStackingAction()
{}

LXeStackingAction::~LXeStackingAction()
{}

G4ClassificationOfNewTrack
LXeStackingAction::ClassifyNewTrack(const G4Track * aTrack){
 
  G4cout << "LXeStackingAction::ClassifyNewTrack TrackID " << aTrack->GetTrackID() << " ParentID " << aTrack->GetParentID() <<  G4endl;   

  G4bool is_op = aTrack->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition() ; 
  G4bool is_secondary = aTrack->GetParentID()>0 ; 

  G4EventManager* evtmgr = G4EventManager::GetEventManager();
  const G4Event* event = evtmgr->GetConstCurrentEvent() ; 
  LXeUserEventInformation* eventInformation = (LXeUserEventInformation*)event->GetUserInformation();
  
  //Count what process generated the optical photons
  if(is_op){ 
      if(is_secondary){
         G4String procname = aTrack->GetCreatorProcess()->GetProcessName() ;
         G4cout << "LXeStackingAction::ClassifyNewTrack OP Secondary from " << procname << G4endl;  

         if(procname=="Scintillation") eventInformation->IncPhotonCount_Scint();
         else if(procname=="Cerenkov") eventInformation->IncPhotonCount_Ceren();
      }
  }
  return fUrgent;
}

void LXeStackingAction::NewStage(){

  G4cout << "LXeStackingAction::NewStage" << G4endl;   

}

void LXeStackingAction::PrepareNewEvent(){ 

  G4cout << "LXeStackingAction::PrepareNewEvent" << G4endl;   


}








