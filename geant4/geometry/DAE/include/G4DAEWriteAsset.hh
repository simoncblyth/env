#ifndef _G4DAEWRITEASSET_INCLUDED_
#define _G4DAEWRITEASSET_INCLUDED_

#include "G4DAEWrite.hh"
#include <sstream>

class G4DAEWriteAsset : public G4DAEWrite
{

 protected:

   G4ThreeVector GetAngles(const G4RotationMatrix&);
   void ScaleWrite(xercesc::DOMElement* element, const G4String& name,
                    const G4ThreeVector& scl)
     { Scale_vectorWrite(element,"scale",name,scl); }
   void RotationWrite(xercesc::DOMElement* element, const G4String& name,
                    const G4ThreeVector& rot)
     { Rotation_vectorWrite(element,"rotation",name,rot); }
   void PositionWrite(xercesc::DOMElement* element, const G4String& name,
                    const G4ThreeVector& pos)
     { Position_vectorWrite(element,"position",name,pos); }
   void FirstrotationWrite(xercesc::DOMElement* element, const G4String& name,
                    const G4ThreeVector& rot)
     { Rotation_vectorWrite(element,"firstrotation",name,rot); }
   void FirstpositionWrite(xercesc::DOMElement* element, const G4String& name,
                    const G4ThreeVector& pos)
     { Position_vectorWrite(element,"firstposition",name,pos); }
   void AddPosition(const G4String& name,
                    const G4ThreeVector& pos)
     { Position_vectorWrite(defineElement,"position",name,pos); }

 protected:

   static const G4double kRelativePrecision;
   static const G4double kAngularPrecision;
   static const G4double kLinearPrecision;
  
 private:

   void Scale_vectorWrite(xercesc::DOMElement*, const G4String&,
                             const G4String&, const G4ThreeVector&);
   void Rotation_vectorWrite(xercesc::DOMElement*, const G4String&,
                             const G4String&, const G4ThreeVector&);
   void Position_vectorWrite(xercesc::DOMElement*, const G4String&,
                             const G4String&, const G4ThreeVector&);


 private:

   void AssetWrite(xercesc::DOMElement*);

 private:

   xercesc::DOMElement* defineElement;
   xercesc::DOMElement* assetElement;

};

#endif
