#include "G4DAEWriteEffects.hh"
#include "G4Color.hh"
#include <sstream>

void G4DAEWriteEffects::
ColorTypeWrite(xercesc::DOMElement* element, const G4String& type, const G4Color& c)
{
   std::ostringstream tc;
   tc << c.GetRed() << " " << c.GetGreen() << " " << c.GetBlue() << " " << c.GetAlpha() ;    
   std::string tcs = tc.str();

   xercesc::DOMElement* typeElement = NewElement(type);  // emission, ambient, diffuse, specular, reflective, transparent
   xercesc::DOMElement* colorElement = NewTextElement("color", tcs.c_str() );
   typeElement->appendChild(colorElement);
   element->appendChild(typeElement);
}

void G4DAEWriteEffects::
FloatTypeWrite(xercesc::DOMElement* element,const G4String& type, const G4double& value )
{
   xercesc::DOMElement* typeElement = NewElement(type);  

   std::ostringstream tc;
   tc << value ; 
   std::string tcs = tc.str();

   xercesc::DOMElement* floatElement = NewTextElement("float", tcs.c_str());
   typeElement->appendChild(floatElement);
   element->appendChild(typeElement);
}


void G4DAEWriteEffects::EffectWrite(const G4Material* const materialPtr )
{
   // NB assuming 1to1 between materials and effects 

   const G4String name = GenerateName(materialPtr->GetName() + "_fx_" , materialPtr) ;
   const G4String& id = name ;
   const G4String& sid = name ;


   xercesc::DOMElement* effectElement = NewElementOneNCNameAtt("effect","id",id);

   /*
   // Problems with textures...
   //
   //    * not supported in MeshLab it seems 
   //    * schema validation issue with non-unique id (probably simple to fix)
   //
   // So postpone this purely visual feature that complicates DAE 
   // usage due to lack of texture supprt.
   //

   G4String matSymbol = GenerateMaterialSymbol(materialPtr->GetName()) ;  
   const G4String img_id = matSymbol  ;
   xercesc::DOMElement* imageElement = NewElementOneNCNameAtt("image","id",img_id);
   xercesc::DOMElement* initfromElement = NewTextElement("init_from", img_id );
   imageElement->appendChild(initfromElement);
   effectElement->appendChild(imageElement);

   // add to the library 
   const G4String texpath = GenerateTexturePath( img_id ); 
   xercesc::DOMElement* libImageElement = NewElementOneNCNameAtt("image","id",img_id);
   xercesc::DOMElement* libInitfromElement = NewTextElement("init_from", texpath );
   libImageElement->appendChild(libInitfromElement);
   imagesElement->appendChild(libImageElement);   // library_images written by EffectsWrite

   */


   xercesc::DOMElement* profileElement = NewElement("profile_COMMON");
   xercesc::DOMElement* techniqueElement = NewElementOneNCNameAtt("technique","sid",sid);
  
   const G4Color& color = G4Color::White();
   G4double shininess = 20. ;
   G4double reflectivity = 0.5 ;
   G4double transparency = 0.5 ;

   xercesc::DOMElement* phongElement = NewElement("phong");
   ColorTypeWrite(phongElement, "emission", color );
   ColorTypeWrite(phongElement, "ambient", color );
   ColorTypeWrite(phongElement, "diffuse", color );
   ColorTypeWrite(phongElement, "specular", color );
   FloatTypeWrite(phongElement, "shininess", shininess );
   ColorTypeWrite(phongElement, "reflective", color );
   FloatTypeWrite(phongElement, "reflectivity", reflectivity );
   ColorTypeWrite(phongElement, "transparent", color );
   FloatTypeWrite(phongElement, "transparency",  transparency );

   techniqueElement->appendChild(phongElement);
   profileElement->appendChild(techniqueElement);
   effectElement->appendChild(profileElement);

   effectsElement->appendChild(effectElement);
     // Append the effect AFTER all the possible components are appended!
}

void G4DAEWriteEffects::EffectsWrite(xercesc::DOMElement* element)
{
   G4cout << "G4DAE: Writing library_effects..." << G4endl;

   effectsElement = NewElement("library_effects");
   element->appendChild(effectsElement);

   effectList.clear();

   /*
   G4cout << "G4DAE: Writing library_images..." << G4endl;

   imagesElement = NewElement("library_images");
   element->appendChild(imagesElement);
   */

}

void G4DAEWriteEffects::AddEffectMaterial(const G4Material* const materialPtr)
{
   for (size_t i=0;i<effectList.size();i++)    // Check if material is
   {                                             // already in the list!
      if (effectList[i] == materialPtr)  { return; }
   }
   effectList.push_back(materialPtr);
   EffectWrite(materialPtr);
}

