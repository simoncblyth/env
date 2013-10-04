#ifndef _G4DAEWRITEPARAMVOL_INCLUDED_
#define _G4DAEWRITEPARAMVOL_INCLUDED_

#include "G4PVParameterised.hh"
#include "G4VPhysicalVolume.hh"

#include "G4DAEWriteSetup.hh"

class G4DAEWriteParamvol : public G4DAEWriteSetup
{

 protected:

   void ParamvolWrite(xercesc::DOMElement*, const G4VPhysicalVolume* const);

 private:

   void Box_dimensionsWrite(xercesc::DOMElement*, const G4Box* const);
   void Trd_dimensionsWrite(xercesc::DOMElement*, const G4Trd* const);
   void Trap_dimensionsWrite(xercesc::DOMElement*, const G4Trap* const);
   void Tube_dimensionsWrite(xercesc::DOMElement*, const G4Tubs* const);
   void Cone_dimensionsWrite(xercesc::DOMElement*, const G4Cons* const);
   void Sphere_dimensionsWrite(xercesc::DOMElement*, const G4Sphere* const);
   void Orb_dimensionsWrite(xercesc::DOMElement*, const G4Orb* const);
   void Torus_dimensionsWrite(xercesc::DOMElement*, const G4Torus* const);
   void Para_dimensionsWrite(xercesc::DOMElement*, const G4Para* const);
   void Hype_dimensionsWrite(xercesc::DOMElement*, const G4Hype* const);
   void ParametersWrite(xercesc::DOMElement*,
                        const G4VPhysicalVolume* const, const G4int&);
   void ParamvolAlgorithmWrite(xercesc::DOMElement* paramvolElement,
                               const G4VPhysicalVolume* const paramvol);

};

#endif
