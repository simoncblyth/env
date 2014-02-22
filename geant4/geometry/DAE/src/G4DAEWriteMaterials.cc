#include "G4DAEWriteMaterials.hh"

void G4DAEWriteMaterials::
AtomWrite(xercesc::DOMElement* element,const G4double& a)
{
   xercesc::DOMElement* atomElement = NewElement("atom");
   atomElement->setAttributeNode(NewAttribute("unit","g/mole"));
   atomElement->setAttributeNode(NewAttribute("value",a*mole/g));
   element->appendChild(atomElement);
}

void G4DAEWriteMaterials::
DWrite(xercesc::DOMElement* element,const G4double& d)
{
   xercesc::DOMElement* DElement = NewElement("D");
   DElement->setAttributeNode(NewAttribute("unit","g/cm3"));
   DElement->setAttributeNode(NewAttribute("value",d*cm3/g));
   element->appendChild(DElement);
}

void G4DAEWriteMaterials::
PWrite(xercesc::DOMElement* element,const G4double& P)
{
   xercesc::DOMElement* PElement = NewElement("P");
   PElement->setAttributeNode(NewAttribute("unit","pascal"));
   PElement->setAttributeNode(NewAttribute("value",P/pascal));
   element->appendChild(PElement);
}

void G4DAEWriteMaterials::
TWrite(xercesc::DOMElement* element,const G4double& T)
{
   xercesc::DOMElement* TElement = NewElement("T");
   TElement->setAttributeNode(NewAttribute("unit","K"));
   TElement->setAttributeNode(NewAttribute("value",T/kelvin));
   element->appendChild(TElement);
}

void G4DAEWriteMaterials::
IsotopeWrite(const G4Isotope* const isotopePtr)
{
   const G4String name = GenerateName(isotopePtr->GetName(),isotopePtr);

   xercesc::DOMElement* isotopeElement = NewElement("isotope");
   isotopeElement->setAttributeNode(NewAttribute("name",name));
   isotopeElement->setAttributeNode(NewAttribute("N",isotopePtr->GetN()));
   isotopeElement->setAttributeNode(NewAttribute("Z",isotopePtr->GetZ()));
   materialsElement->appendChild(isotopeElement);
   AtomWrite(isotopeElement,isotopePtr->GetA());
}

void G4DAEWriteMaterials::ElementWrite(const G4Element* const elementPtr)
{
   const G4String name = GenerateName(elementPtr->GetName(),elementPtr);

   xercesc::DOMElement* elementElement = NewElement("element");
   elementElement->setAttributeNode(NewAttribute("name",name));

   const size_t NumberOfIsotopes = elementPtr->GetNumberOfIsotopes();

   if (NumberOfIsotopes>0)
   {
      const G4double* RelativeAbundanceVector =
            elementPtr->GetRelativeAbundanceVector();             
      for (size_t i=0;i<NumberOfIsotopes;i++)
      {
         G4String fractionref = GenerateName(elementPtr->GetIsotope(i)->GetName(),
                                             elementPtr->GetIsotope(i));
         xercesc::DOMElement* fractionElement = NewElement("fraction");
         fractionElement->setAttributeNode(NewAttribute("n",
                                           RelativeAbundanceVector[i]));
         fractionElement->setAttributeNode(NewAttribute("ref",fractionref));
         elementElement->appendChild(fractionElement);
         AddIsotope(elementPtr->GetIsotope(i));
      }
   }
   else
   {
      elementElement->setAttributeNode(NewAttribute("Z",elementPtr->GetZ()));
      AtomWrite(elementElement,elementPtr->GetA());
   }

   materialsElement->appendChild(elementElement);
     // Append the element AFTER all the possible components are appended!
}

void G4DAEWriteMaterials::MaterialWrite(const G4Material* const materialPtr)
{
   const G4String matname = GenerateName(materialPtr->GetName(), materialPtr);
   const G4String fxname = GenerateName(materialPtr->GetName() + "_fx_", materialPtr);

   xercesc::DOMElement* materialElement = NewElementOneNCNameAtt("material","id",matname);
   xercesc::DOMElement* instanceEffectElement = NewElementOneNCNameAtt("instance_effect","url",fxname, true);
   materialElement->appendChild(instanceEffectElement);

   G4MaterialPropertiesTable* ptable = materialPtr->GetMaterialPropertiesTable();
   if(ptable)
   {   
       xercesc::DOMElement* extraElement = NewElement("extra");
       PropertyWrite(extraElement, ptable);
       materialElement->appendChild(extraElement);
   }  
 
   materialsElement->appendChild(materialElement);

     // Append the material AFTER all the possible components are appended!
}



void G4DAEWriteMaterials::MaterialsWrite(xercesc::DOMElement* element)
{
   G4cout << "G4DAE: Writing library_materials..." << G4endl;

   materialsElement = NewElement("library_materials");
   element->appendChild(materialsElement);

   isotopeList.clear();
   elementList.clear();
   materialList.clear();

}

void G4DAEWriteMaterials::AddIsotope(const G4Isotope* const isotopePtr)
{
   for (size_t i=0; i<isotopeList.size(); i++)   // Check if isotope is
   {                                             // already in the list!
     if (isotopeList[i] == isotopePtr)  { return; }
   }
   isotopeList.push_back(isotopePtr);
   IsotopeWrite(isotopePtr);
}

void G4DAEWriteMaterials::AddElement(const G4Element* const elementPtr)
{
   for (size_t i=0;i<elementList.size();i++)     // Check if element is
   {                                             // already in the list!
      if (elementList[i] == elementPtr) { return; }
   }
   elementList.push_back(elementPtr);
   ElementWrite(elementPtr);
}

void G4DAEWriteMaterials::AddMaterial(const G4Material* const materialPtr)
{
   for (size_t i=0;i<materialList.size();i++)    // Check if material is
   {                                             // already in the list!
      if (materialList[i] == materialPtr)  { return; }
   }
   materialList.push_back(materialPtr);
   MaterialWrite(materialPtr);
}
