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
//
// $Id: geotest.cc,v 1.1.2.1 2009/03/03 13:50:54 gcosmo Exp $
// GEANT4 tag $Name: geant4-09-02-patch-01 $
//
//
// --------------------------------------------------------------
//      GEANT 4 - geotest
//
// --------------------------------------------------------------

// Geant4 includes
//
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4UItcsh.hh"
#include "G4VisExecutive.hh"
#include "globals.hh"

// A pre-built physics list
//
#include "QGSP_EMV.hh"

// Example includes
//
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"


int main(int argc, char** argv)
{
       
  // Construct the default run manager
  //
  G4RunManager* runManager = new G4RunManager;
  G4VisManager* visManager = new G4VisExecutive;
        

  // Set mandatory initialization and user action classes
  //
  DetectorConstruction* detector = new DetectorConstruction;
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(new QGSP_EMV);
  runManager->SetUserAction(new PrimaryGeneratorAction);
  RunAction* runAction = new RunAction;
  runManager->SetUserAction(runAction);
      
  // Initialisation of runManager via macro for the interactive mode
  // This gives possibility to give different names for GDML file to READ
 
  visManager->Initialize();

  // Open a UI session: will stay there until the user types "exit"
  //
  G4UIsession* session = 0;

  if ( argc==1 )   // Automatically run default macro for writing... 
  {

#ifdef G4UI_USE_TCSH
    session = new G4UIterminal(new G4UItcsh);
#else
    session = new G4UIterminal();
#endif
    G4UImanager* UI = G4UImanager::GetUIpointer(); 

    UI->ApplyCommand("/control/execute vis.mac");

    session->SessionStart();
  }
  else             // Provides macro in input
  { 
#ifdef G4UI_USE_TCSH
    session = new G4UIterminal(new G4UItcsh);
#else
    session = new G4UIterminal();
#endif
    G4String command = "/control/execute "; 
    G4String fileName = argv[1]; 
    G4UImanager* UI = G4UImanager::GetUIpointer(); 
    UI->ApplyCommand(command+fileName); 
    session->SessionStart();
  }
  
  delete session;
  delete visManager;
  
  // Job termination
  //
  delete runManager;

  return 0;
}
