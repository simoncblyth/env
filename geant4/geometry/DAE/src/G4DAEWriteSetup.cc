#include "G4DAEWriteSetup.hh"

void G4DAEWriteSetup::SetupWrite(xercesc::DOMElement* daeElement,
                                   const G4LogicalVolume* const logvol)
{
   G4cout << "G4DAE: Writing library_visual_scenes..." << G4endl;

   xercesc::DOMElement* lvsElement = NewElement("library_visual_scenes");
   xercesc::DOMElement* vsElement = NewElementOneAtt("visual_scene","id","DefaultScene");

   const G4String& lvname = GenerateName(logvol->GetName(),logvol);

   xercesc::DOMElement* noElement = NewElementOneAtt("node","id","top");
   xercesc::DOMElement* niElement = NewElementOneNCNameAtt("instance_node","url", lvname, true);
   noElement->appendChild(niElement);
   vsElement->appendChild(noElement);
   lvsElement->appendChild(vsElement);
   daeElement->appendChild(lvsElement);

   // ---------  
 
   xercesc::DOMElement* sceneElement = NewElement("scene");
   xercesc::DOMElement* ivsElement = NewElementOneAtt("instance_visual_scene","url","#DefaultScene");
   sceneElement->appendChild(ivsElement);
   daeElement->appendChild(sceneElement);

}
