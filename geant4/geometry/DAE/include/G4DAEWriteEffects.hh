#ifndef _G4DAEWRITEEFFECTS_INCLUDED_
#define _G4DAEWRITEEFFECTS_INCLUDED_

#include "G4Material.hh"
#include "G4DAEWriteAsset.hh"
#include "G4Color.hh"


class G4DAEWriteEffects : public G4DAEWriteAsset
{
 protected:
   void AddEffectMaterial(const G4Material* const materialPtr);

 private:
   void ColorTypeWrite(xercesc::DOMElement* element, const G4String&, const G4Color& );
   void FloatTypeWrite(xercesc::DOMElement* element, const G4String&, const G4double& );
   void EffectWrite(const G4Material* const materialPtr);
   void EffectsWrite(xercesc::DOMElement*);

 private:
   std::vector<const G4Material*> effectList;
   xercesc::DOMElement* effectsElement;
   xercesc::DOMElement* imagesElement;

};

#endif
