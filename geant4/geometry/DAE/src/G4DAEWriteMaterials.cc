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

   xercesc::DOMElement* extraElement = NewElement("extra");
   materialElement->appendChild(extraElement);

   if (materialPtr->GetMaterialPropertiesTable())
   {   
       PropertyWrite(materialElement, materialPtr, extraElement);
   }   
   materialsElement->appendChild(materialElement);

     // Append the material AFTER all the possible components are appended!
}

// adapted from /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLWriteMaterials.cc
// needs access to property map, so must patch older geant4 to have access to the map


void G4DAEWriteMaterials::PropertyVectorWrite(const G4String& key,
                           const G4MaterialPropertyVector* const pvec, 
                           //const G4PhysicsOrderedFreeVector* const pvec, 
                            xercesc::DOMElement* extraElement)
{
   const G4String matrixref = GenerateName(key, pvec);
   xercesc::DOMElement* matrixElement = NewElement("matrix");
   matrixElement->setAttributeNode(NewAttribute("name", matrixref));
   matrixElement->setAttributeNode(NewAttribute("coldim", "2"));
   std::ostringstream pvalues;
   for (size_t i=0; i<pvec->GetVectorLength(); i++)
   {
       if (i!=0)  { pvalues << " "; }
       pvalues << pvec->GetLowEdgeEnergy(i) << " " << (*pvec)[i];
       //
       // pvec->GetLowEdgeEnergy(i) is tentative translation 
       // from future Geant4 Energy(i)
       //
   }
   matrixElement->setAttributeNode(NewAttribute("values", pvalues.str()));

   extraElement->appendChild(matrixElement);  // was toplevel defineElement for GDML
}



void G4DAEWriteMaterials::PropertyWrite(xercesc::DOMElement* matElement,
                                         const G4Material* const mat,
                                        xercesc::DOMElement* extraElement)
{
   xercesc::DOMElement* propElement;
   G4MaterialPropertiesTable* ptable = mat->GetMaterialPropertiesTable();
   const std::map< G4String, G4MaterialPropertyVector*,
                 std::less<G4String> >* pmap = ptable->GetPropertiesMap();
   const std::map< G4String, G4double,
                 std::less<G4String> >* cmap = ptable->GetPropertiesCMap();
   std::map< G4String, G4MaterialPropertyVector*,
                 std::less<G4String> >::const_iterator mpos;
   std::map< G4String, G4double,
                 std::less<G4String> >::const_iterator cpos;
   for (mpos=pmap->begin(); mpos!=pmap->end(); mpos++)
   {
      propElement = NewElement("property");
      propElement->setAttributeNode(NewAttribute("name", mpos->first));
      propElement->setAttributeNode(NewAttribute("ref",
                                    GenerateName(mpos->first, mpos->second)));
      if (mpos->second)
      {
         PropertyVectorWrite(mpos->first, mpos->second, extraElement);
         extraElement->appendChild(propElement);
      }
      else
      {
         G4String warn_message = "Null pointer for material property -"
                  + mpos->first + "- of material -" + mat->GetName() + "- !";
         G4Exception("G4DAEWriteMaterials::PropertyWrite()", "NullPointer",
                     JustWarning, warn_message);
         continue;
      }
   }
   for (cpos=cmap->begin(); cpos!=cmap->end(); cpos++)
   {
      propElement = NewElement("property");
      propElement->setAttributeNode(NewAttribute("name", cpos->first));
      propElement->setAttributeNode(NewAttribute("ref", cpos->first));
      xercesc::DOMElement* constElement = NewElement("constant");
      constElement->setAttributeNode(NewAttribute("name", cpos->first));
      constElement->setAttributeNode(NewAttribute("value", cpos->second));
      // tacking onto a separate top level define element for GDML
      // but that would need separate access on reading 

      //defineElement->appendChild(constElement);
      extraElement->appendChild(constElement);
      extraElement->appendChild(propElement);
   }
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
