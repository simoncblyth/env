#include "G4DAEWriteAsset.hh"


const G4double G4DAEWriteAsset::kRelativePrecision = DBL_EPSILON;
const G4double G4DAEWriteAsset::kAngularPrecision = DBL_EPSILON;
const G4double G4DAEWriteAsset::kLinearPrecision = DBL_EPSILON;

G4ThreeVector G4DAEWriteAsset::GetAngles(const G4RotationMatrix& mat)
{
   G4double x,y,z;

   const G4double cosb = std::sqrt(mat.xx()*mat.xx()+mat.yx()*mat.yx());

   if (cosb > kRelativePrecision)
   {
      x = std::atan2(mat.zy(),mat.zz());
      y = std::atan2(-mat.zx(),cosb);
      z = std::atan2(mat.yx(),mat.xx());
   }
   else
   {
      x = std::atan2(-mat.yz(),mat.yy());
      y = std::atan2(-mat.zx(),cosb);
      z = 0.0;
   }

   return G4ThreeVector(x,y,z);
}

void G4DAEWriteAsset::
Scale_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
                  const G4String& name, const G4ThreeVector& scl)
{
   const G4double x = (std::fabs(scl.x()-1.0) < kRelativePrecision)
                    ? 1.0 : scl.x();
   const G4double y = (std::fabs(scl.y()-1.0) < kRelativePrecision)
                    ? 1.0 : scl.y();
   const G4double z = (std::fabs(scl.z()-1.0) < kRelativePrecision)
                    ? 1.0 : scl.z();

   xercesc::DOMElement* scaleElement = NewElement(tag);
   scaleElement->setAttributeNode(NewAttribute("name",name));
   scaleElement->setAttributeNode(NewAttribute("x",x));
   scaleElement->setAttributeNode(NewAttribute("y",y));
   scaleElement->setAttributeNode(NewAttribute("z",z));
   element->appendChild(scaleElement);
}

void G4DAEWriteAsset::
Rotation_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
                     const G4String& name, const G4ThreeVector& rot)
{
   const G4double x = (std::fabs(rot.x()) < kAngularPrecision) ? 0.0 : rot.x();
   const G4double y = (std::fabs(rot.y()) < kAngularPrecision) ? 0.0 : rot.y();
   const G4double z = (std::fabs(rot.z()) < kAngularPrecision) ? 0.0 : rot.z();

   xercesc::DOMElement* rotationElement = NewElement(tag);
   rotationElement->setAttributeNode(NewAttribute("name",name));
   rotationElement->setAttributeNode(NewAttribute("x",x/degree));
   rotationElement->setAttributeNode(NewAttribute("y",y/degree));
   rotationElement->setAttributeNode(NewAttribute("z",z/degree));
   rotationElement->setAttributeNode(NewAttribute("unit","deg"));
   element->appendChild(rotationElement);
}

void G4DAEWriteAsset::
Position_vectorWrite(xercesc::DOMElement* element, const G4String& tag,
                     const G4String& name, const G4ThreeVector& pos)
{
   const G4double x = (std::fabs(pos.x()) < kLinearPrecision) ? 0.0 : pos.x();
   const G4double y = (std::fabs(pos.y()) < kLinearPrecision) ? 0.0 : pos.y();
   const G4double z = (std::fabs(pos.z()) < kLinearPrecision) ? 0.0 : pos.z();

   xercesc::DOMElement* positionElement = NewElement(tag);
   positionElement->setAttributeNode(NewAttribute("name",name));
   positionElement->setAttributeNode(NewAttribute("x",x/mm));
   positionElement->setAttributeNode(NewAttribute("y",y/mm));
   positionElement->setAttributeNode(NewAttribute("z",z/mm));
   positionElement->setAttributeNode(NewAttribute("unit","mm"));
   element->appendChild(positionElement);
}




void G4DAEWriteAsset::AssetWrite(xercesc::DOMElement* element)
{
   G4cout << "G4DAE: Writing asset metadata..." << G4endl;

   //defineElement = NewElement("define");
   //element->appendChild(defineElement);

   G4String created = "2005-11-14T02:16:38Z" ;
   G4String modified = "2005-11-14T02:16:38Z" ;
   G4String revision = "1.0" ;

   assetElement = NewElement("asset");
   assetElement->appendChild(NewTextElement("created", created));
   assetElement->appendChild(NewTextElement("modified", modified));
   assetElement->appendChild(NewTextElement("revision", revision));

   element->appendChild(assetElement);
   

}
