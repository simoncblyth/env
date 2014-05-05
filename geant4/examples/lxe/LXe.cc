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
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4String.hh"
#include "G4UItcsh.hh"
#include "G4UIterminal.hh"

#include "LXeDetectorConstruction.hh"
#include "LXePhysicsList.hh"
#include "LXePrimaryGeneratorAction.hh"
#include "LXeEventAction.hh"
#include "LXeStackingAction.hh"
#include "LXeSteppingAction.hh"
#include "LXeTrackingAction.hh"
#include "LXeRunAction.hh"
#include "LXeSteppingVerbose.hh"

#include "RecorderBase.hh"

#ifdef G4VIS_USE
#include "G4VisExecutive.hh"
#endif



#ifdef EXPORT_G4DAE

#include "G4DAEParser.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"

#endif



//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
int main(int argc, char** argv)
{
  G4VSteppingVerbose::SetInstance(new LXeSteppingVerbose);

  G4RunManager* runManager = new G4RunManager;

  runManager->SetUserInitialization(new LXeDetectorConstruction);
  runManager->SetUserInitialization(new LXePhysicsList);

#ifdef G4VIS_USE
  G4VisManager* visManager = new G4VisExecutive;
  visManager->Initialize();
#endif

  RecorderBase* recorder = NULL;//No recording is done in this example

  runManager->SetUserAction(new LXePrimaryGeneratorAction);
  runManager->SetUserAction(new LXeStackingAction);
  
  runManager->SetUserAction(new LXeRunAction(recorder));
  runManager->SetUserAction(new LXeEventAction(recorder));
  runManager->SetUserAction(new LXeTrackingAction(recorder));
  runManager->SetUserAction(new LXeSteppingAction(recorder));

  runManager->Initialize();

#ifdef EXPORT_G4DAE
  G4cout << "Exporting geometry due to  -DEXPORT_G4DAE  " << G4endl ; 
  G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
      GetNavigatorForTracking()->GetWorldVolume();

  G4DAEParser dae ; 

  G4String path = "g4_00.dae" ; 
  G4bool refs = true ;
  G4bool recreatePoly = false ; 
  G4int nodeIndex = -1 ;            // so World is volume 0 

  dae.Write(path, wpv, refs, recreatePoly, nodeIndex );

#else
  G4cout << "compile with -DEXPORT_G4DAE to export geometry " << G4endl ; 
#endif
 
  // get the pointer to the UI manager and set verbosities
  G4UImanager* UI = G4UImanager::GetUIpointer();
  
  if(argc==1){
    G4UIsession* session=0;
#ifdef G4UI_USE_TCSH
    session = new G4UIterminal(new G4UItcsh);
#else
    session = new G4UIterminal();
#endif

    //execute vis.mac
    UI->ApplyCommand("/control/execute vis.mac");

    session->SessionStart();
    delete session;

  }
  else{
    G4String command = "/control/execute ";
    G4String filename = argv[1];
    UI->ApplyCommand(command+filename);
  }

  if(recorder)delete recorder;

#ifdef G4VIS_USE
  delete visManager;
#endif

  // job termination
  delete runManager;
  return 0;
}


