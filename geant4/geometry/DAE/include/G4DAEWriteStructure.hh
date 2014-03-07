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
class G4VisAttributes ; 

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4PVDivision;
class G4LogicalBorderSurface;
class G4LogicalSkinSurface;
class G4OpticalSurface;
class G4SurfaceProperty;


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

   void MatrixWrite(xercesc::DOMElement* nodeElement, const G4Transform3D& T);

   // from g4.10 G4GDMLWriteStructure 
   void SurfacesWrite();
   void BorderSurfaceCache(const G4LogicalBorderSurface* const);
   void SkinSurfaceCache(const G4LogicalSkinSurface* const);
   const G4LogicalBorderSurface* GetBorderSurface(const G4VPhysicalVolume* const);
   const G4LogicalSkinSurface* GetSkinSurface(const G4LogicalVolume* const);
   G4bool FindOpticalSurface(const G4SurfaceProperty*);

   // over from g4.10 G4GDMLWriteSolids
   void OpticalSurfaceWrite(xercesc::DOMElement*, const G4OpticalSurface* const);

   // debug ambiguous PV1, PV2 
   xercesc::DOMElement* GetBorderSurfacesMetaElement();
   xercesc::DOMElement* GetPVElement(const G4VPhysicalVolume* const pvol);


 private:

   static const int maxReflections = 8; // Constant for limiting the number
                                        // of displacements/reflections applied
                                        // to a single solid

 protected:

   xercesc::DOMElement* structureElement;   // library_nodes
   xercesc::DOMElement* extrasurfElement;   // extra

   std::vector<xercesc::DOMElement*> borderElementVec;
   std::vector<xercesc::DOMElement*> skinElementVec;

 private:  // cache for optical surfaces...

   std::vector<const G4OpticalSurface*> opt_vec;

};

#endif
