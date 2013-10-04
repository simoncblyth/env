#ifndef _G4DAEWRITESTRUCTURE_INCLUDED_
#define _G4DAEWRITESTRUCTURE_INCLUDED_

#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4PVDivision.hh"
#include "G4PVReplica.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ReflectedSolid.hh"
#include "G4Transform3D.hh"

#include "G4DAEWriteParamvol.hh"

class G4DAEWriteStructure : public G4DAEWriteParamvol
{

 private:

   void DivisionvolWrite(xercesc::DOMElement*,const G4PVDivision* const);
   void PhysvolWrite(xercesc::DOMElement*,const G4VPhysicalVolume* const topVol,
                                          const G4Transform3D& transform,
                                          const G4String& moduleName);
   void ReplicavolWrite(xercesc::DOMElement*,const G4VPhysicalVolume* const);
   void StructureWrite(xercesc::DOMElement*);
   G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const topVol,
                                    const G4int depth);
 private:

   xercesc::DOMElement* structureElement;
   static const int maxReflections = 8; // Constant for limiting the number
                                        // of displacements/reflections applied
                                        // to a single solid
};

#endif
