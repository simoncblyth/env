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
// $Id: DetectorMessenger.hh,v 1.1 2008/08/27 10:30:18 gcosmo Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
// Class DetectorMessenger
//
// Utility messenger for defining run-time commands relative to the example.
//
// ----------------------------------------------------------------------------

#ifndef DetectorMessenger_h
#define DetectorMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class DetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;

// ----------------------------------------------------------------------------

class DetectorMessenger: public G4UImessenger
{

  public:

    DetectorMessenger( DetectorConstruction* );
   ~DetectorMessenger();
    
    void SetNewValue( G4UIcommand*, G4String );
    void SetNewValue( G4UIcommand*, G4int );  

  private:

    DetectorConstruction*      theDetector;
    G4UIdirectory*             theDetectorDir;
    G4UIcmdWithAString*        theReadCommand;
    G4UIcmdWithAString*        theWriteCommand;
    G4UIcmdWithAString*        theStepCommand;
};

// ----------------------------------------------------------------------------

#endif
