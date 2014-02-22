#ifndef _G4DAEWRITEMATERIALS_INCLUDED_
#define _G4DAEWRITEMATERIALS_INCLUDED_

#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4Material.hh"
#include "G4String.hh"
#include "G4DAEWriteEffects.hh"

class G4DAEWriteMaterials : public G4DAEWriteEffects
{

 protected:

   void AddIsotope(const G4Isotope* const);
   void AddElement(const G4Element* const);
   void AddMaterial(const G4Material* const);

 private:

   void AtomWrite(xercesc::DOMElement*,const G4double&);
   void DWrite(xercesc::DOMElement*,const G4double&);
   void PWrite(xercesc::DOMElement*,const G4double&);
   void TWrite(xercesc::DOMElement*,const G4double&);
   void IsotopeWrite(const G4Isotope* const);
   void ElementWrite(const G4Element* const);
   void MaterialWrite(const G4Material* const);
   void MaterialsWrite(xercesc::DOMElement*);



 private:

   std::vector<const G4Isotope*> isotopeList;
   std::vector<const G4Element*> elementList;
   std::vector<const G4Material*> materialList;

   xercesc::DOMElement* materialsElement;
};

#endif
