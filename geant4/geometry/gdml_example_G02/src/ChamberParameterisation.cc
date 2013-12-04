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
// $Id: ChamberParameterisation.cc,v 1.2 2008/12/18 12:57:02 gunter Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
// Class ChamberParameterisation implementation
//
// ----------------------------------------------------------------------------

#include "ChamberParameterisation.hh"

#include "G4ThreeVector.hh"
#include "G4Box.hh"
#include "G4VPhysicalVolume.hh"

// ----------------------------------------------------------------------------

ChamberParameterisation::
ChamberParameterisation( G4int    NoChambers, 
                         G4double startZ,          //  Z of center of first 
                         G4double spacingZ,        //  Z spacing of centers
                         G4double widthChamber, 
                         G4double lengthInitial, 
                         G4double lengthFinal )
{
  fNoChambers =  NoChambers; 
  fStartZ     =  startZ; 
  fHalfWidth  =  widthChamber*0.5;
  fSpacing    =  spacingZ;
  fHalfLengthFirst = 0.5 * lengthInitial; 

  if( NoChambers > 0 )
  {
    fHalfLengthIncr = 0.5 * (lengthFinal-lengthInitial)/NoChambers;
    if (spacingZ < widthChamber)
    {
      G4Exception("ExN02ChamberParameterisation::ChamberParameterisation()",
                  "InvalidSetup", FatalException,
                  "Invalid construction: Width>Spacing");
    }
  }
}

// ----------------------------------------------------------------------------

ChamberParameterisation::~ChamberParameterisation()
{}

// ----------------------------------------------------------------------------

void ChamberParameterisation::
ComputeTransformation (const G4int copyNo, G4VPhysicalVolume* physVol) const
{
  G4double Zposition = fStartZ + (copyNo+1) * fSpacing;
  G4ThreeVector origin(0.,0.,Zposition);
  physVol->SetTranslation(origin);
  physVol->SetRotation(0);
}

// ----------------------------------------------------------------------------

void ChamberParameterisation::
ComputeDimensions (G4Box& trackerChamber, const G4int copyNo,
                   const G4VPhysicalVolume*) const
{
  G4double halfLength = fHalfLengthFirst + copyNo * fHalfLengthIncr;
  trackerChamber.SetXHalfLength(halfLength);
  trackerChamber.SetYHalfLength(halfLength);
  trackerChamber.SetZHalfLength(fHalfWidth);
}

// ----------------------------------------------------------------------------
