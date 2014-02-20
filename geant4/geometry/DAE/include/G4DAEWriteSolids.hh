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

   void AddSolid(const G4VSolid* const, const G4String& matSymbol );

 private:

   void AccessorWrite(xercesc::DOMElement*, const G4String&, G4int, G4int, const G4String& vars );
   void InputWrite(xercesc::DOMElement*, const G4String&, const G4String&, G4int offset);
   G4String FloatArrayWrite(xercesc::DOMElement* srcElement, const G4String& srcId, G4int count, const G4String& data);
   G4String SourceWrite(xercesc::DOMElement* meshElement, const G4String& geoId, const G4String& ext, G4int items, G4int stride, const G4String& data);
   G4String VerticesWrite(xercesc::DOMElement* meshElement, const G4String& geoId, const G4String& ext, const G4String& posRef);
   void PolygonsWrite(xercesc::DOMElement* meshElement, const G4String& vtxRef, const G4String& nrmRef, std::vector<std::string>& facets, const G4String& material);
   void PolylistWrite(xercesc::DOMElement* meshElement, const G4String& vtxRef, const G4String& nrmRef, std::vector<std::string>& facets, std::vector<std::string>& vcount, const G4String& material, const G4String& texRef );
   G4String GeometryWrite(xercesc::DOMElement* solidsElement, const G4VSolid* const solid, const G4String& matSymbol );
   void MetadataWrite(xercesc::DOMElement* meshElement, const G4String& geoId, std::map<std::string,std::string>& meta );


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

  protected:

   std::vector<const G4VSolid*> solidList;
   xercesc::DOMElement* solidsElement;




};

#endif
