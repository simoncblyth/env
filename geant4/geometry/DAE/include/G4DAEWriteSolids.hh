#ifndef _G4DAEWRITESOLIDS_INCLUDED_
#define _G4DAEWRITESOLIDS_INCLUDED_

#include "G4BooleanSolid.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Ellipsoid.hh"
#include "G4EllipticalCone.hh"
#include "G4EllipticalTube.hh"
#include "G4ExtrudedSolid.hh"
#include "G4Hype.hh"
#include "G4Orb.hh"
#include "G4Para.hh"
#include "G4Paraboloid.hh"
#include "G4IntersectionSolid.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
#include "G4ReflectedSolid.hh"
#include "G4Sphere.hh"
#include "G4SubtractionSolid.hh"
#include "G4TessellatedSolid.hh"
#include "G4Tet.hh"
#include "G4Torus.hh"
#include "G4Trap.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4TwistedBox.hh"
#include "G4TwistedTrap.hh"
#include "G4TwistedTrd.hh"
#include "G4TwistedTubs.hh"
#include "G4UnionSolid.hh"

#include "G4DAEWriteMaterials.hh"

class G4DAEWriteSolids : public G4DAEWriteMaterials
{

 protected:

   void AddSolid(const G4VSolid* const);

 private:

   void AccessorXYZWrite(xercesc::DOMElement*, const G4String&, G4int, G4int );
   void SourceVerticesWrite(xercesc::DOMElement*, const G4VSolid* const);
   void MeshWrite(xercesc::DOMElement*, const G4VSolid* const);

   void BooleanWrite(xercesc::DOMElement*, const G4BooleanSolid* const);
   void BoxWrite(xercesc::DOMElement*, const G4Box* const);
   void ConeWrite(xercesc::DOMElement*, const G4Cons* const);
   void ElconeWrite(xercesc::DOMElement*, const G4EllipticalCone* const);
   void EllipsoidWrite(xercesc::DOMElement*, const G4Ellipsoid* const);
   void EltubeWrite(xercesc::DOMElement*, const G4EllipticalTube* const);
   void XtruWrite(xercesc::DOMElement*, const G4ExtrudedSolid* const);
   void HypeWrite(xercesc::DOMElement*, const G4Hype* const);
   void OrbWrite(xercesc::DOMElement*, const G4Orb* const);
   void ParaWrite(xercesc::DOMElement*, const G4Para* const);
   void ParaboloidWrite(xercesc::DOMElement*, const G4Paraboloid* const);
   void PolyconeWrite(xercesc::DOMElement*, const G4Polycone* const);
   void PolyhedraWrite(xercesc::DOMElement*, const G4Polyhedra* const);
   void SphereWrite(xercesc::DOMElement*, const G4Sphere* const);
   void TessellatedWrite(xercesc::DOMElement*, const G4TessellatedSolid* const);
   void TetWrite(xercesc::DOMElement*, const G4Tet* const);
   void TorusWrite(xercesc::DOMElement*, const G4Torus* const);
   void TrapWrite(xercesc::DOMElement*, const G4Trap* const);
   void TrdWrite(xercesc::DOMElement*, const G4Trd* const);
   void TubeWrite(xercesc::DOMElement*, const G4Tubs* const);
   void TwistedboxWrite(xercesc::DOMElement*, const G4TwistedBox* const);
   void TwistedtrapWrite(xercesc::DOMElement*, const G4TwistedTrap* const);
   void TwistedtrdWrite(xercesc::DOMElement*, const G4TwistedTrd* const);
   void TwistedtubsWrite(xercesc::DOMElement*, const G4TwistedTubs* const);
   void ZplaneWrite(xercesc::DOMElement*, const G4double&,
                    const G4double&, const G4double&);
   void SolidsWrite(xercesc::DOMElement*);

 private:

   std::vector<const G4VSolid*> solidList;
   xercesc::DOMElement* solidsElement;
};

#endif
