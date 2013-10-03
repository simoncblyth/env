#include "G4DAEWrite.hh"

G4bool G4DAEWrite::addPointerToName = true;

G4bool G4DAEWrite::FileExists(const G4String& fname) const
{
  struct stat FileInfo;
  return (stat(fname.c_str(),&FileInfo) == 0); 
}

G4DAEWrite::VolumeMapType& G4DAEWrite::VolumeMap()
{
   static VolumeMapType instance;
   return instance;
}

G4DAEWrite::PhysVolumeMapType& G4DAEWrite::PvolumeMap()
{
   static PhysVolumeMapType instance;
   return instance;
}

G4DAEWrite::DepthMapType& G4DAEWrite::DepthMap()
{
   static DepthMapType instance;
   return instance;
}

G4String G4DAEWrite::GenerateName(const G4String& name, const void* const ptr)
{
   G4String nameOut;
   std::stringstream stream; stream << name;
   if (addPointerToName) { stream << ptr; };

   nameOut=G4String(stream.str());
   if(nameOut.contains(' '))
   nameOut.erase(std::remove(nameOut.begin(),nameOut.end(),' '),nameOut.end());

   return nameOut;
}

xercesc::DOMAttr* G4DAEWrite::NewAttribute(const G4String& name,
                                            const G4String& value)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
   xercesc::XMLString::transcode(value,tempStr,tempStrSize-1);
   att->setValue(tempStr);
   return att;
}

xercesc::DOMAttr* G4DAEWrite::NewAttribute(const G4String& name,
                                            const G4double& value)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   xercesc::DOMAttr* att = doc->createAttribute(tempStr);
   std::ostringstream ostream;
   ostream.precision(15);
   ostream << value;
   G4String str = ostream.str();
   xercesc::XMLString::transcode(str,tempStr,tempStrSize-1);
   att->setValue(tempStr);
   return att;
}

xercesc::DOMElement* G4DAEWrite::NewElement(const G4String& name)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   return doc->createElement(tempStr);
}


xercesc::DOMElement* G4DAEWrite::NewTextElement(const G4String& name, const G4String& text)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   xercesc::DOMElement* e = doc->createElement(tempStr);
   xercesc::XMLString::transcode(text,tempStr,tempStrSize-1);
   e->setTextContent(tempStr);
   return e; 
}




G4Transform3D G4DAEWrite::Write(const G4String& fname,
                                 const G4LogicalVolume* const logvol,
                                 const G4String& setSchemaLocation,
                                 const G4int depth,
                                       G4bool refs)
{
   SchemaLocation = setSchemaLocation;
   addPointerToName = refs;

   if (depth==0) { G4cout << "G4DAE: Writing '" << fname << "'..." << G4endl; }
   else   { G4cout << "G4DAE: Writing module '" << fname << "'..." << G4endl; }
   
   if (FileExists(fname))
   {
     G4String ErrorMessage = "File '"+fname+"' already exists!";
     G4Exception("G4DAEWrite::Write()", "InvalidSetup",
                 FatalException, ErrorMessage);
   }
   
   VolumeMap().clear(); // The module map is global for all modules,
                        // so clear it only at once!

   xercesc::XMLString::transcode("LS", tempStr, tempStrSize-1);
   xercesc::DOMImplementation* impl =
     xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("Range", tempStr, tempStrSize-1);
   impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
   xercesc::XMLString::transcode("COLLADA", tempStr, tempStrSize-1);
   doc = impl->createDocument(0,tempStr,0);
   xercesc::DOMElement* dae = doc->getDocumentElement();

#if XERCES_VERSION_MAJOR >= 3
                                             // DOM L3 as per Xerces 3.0 API
    xercesc::DOMLSSerializer* writer =
      ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();

    xercesc::DOMConfiguration *dc = writer->getDomConfig();
    dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#else

   xercesc::DOMWriter* writer =
     ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();

   if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
       writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#endif

   dae->setAttributeNode(NewAttribute("xmlns",
                          "http://www.collada.org/2005/11/COLLADASchema"));
   dae->setAttributeNode(NewAttribute("version","1.4.1"));

   //dae->setAttributeNode(NewAttribute("xmlns",
   //                       "http://www.collada.org/2008/03/COLLADASchema");
   //dae->setAttributeNode(NewAttribute("version","1.5.0"));


   AssetWrite(dae);
   EffectsWrite(dae);
   MaterialsWrite(dae);
   SolidsWrite(dae);
   StructureWrite(dae);
   SetupWrite(dae,logvol);

   G4Transform3D R = TraverseVolumeTree(logvol,depth);

   xercesc::XMLFormatTarget *myFormTarget =
     new xercesc::LocalFileFormatTarget(fname.c_str());

   try
   {
#if XERCES_VERSION_MAJOR >= 3
                                            // DOM L3 as per Xerces 3.0 API
      xercesc::DOMLSOutput *theOutput =
        ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
      theOutput->setByteStream(myFormTarget);
      writer->write(doc, theOutput);
#else
      writer->writeNode(myFormTarget, *doc);
#endif
   }
   catch (const xercesc::XMLException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.getMessage());
      G4cout << "G4DAE: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
      return G4Transform3D::Identity;
   }
   catch (const xercesc::DOMException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      G4cout << "G4DAE: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
      return G4Transform3D::Identity;
   }
   catch (...)
   {   
      G4cout << "G4DAE: Unexpected Exception!" << G4endl;
      return G4Transform3D::Identity;
   }        

   delete myFormTarget;
   writer->release();

   if (depth==0)
   {
     G4cout << "G4DAE: Writing '" << fname << "' done !" << G4endl;
   }
   else
   {
     G4cout << "G4DAE: Writing module '" << fname << "' done !" << G4endl;
   }

   return R;
}

void G4DAEWrite::AddModule(const G4VPhysicalVolume* const physvol)
{
   G4String fname = GenerateName(physvol->GetName(),physvol);
   G4cout << "G4DAE: Adding module '" << fname << "'..." << G4endl;

   if (physvol == 0)
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "Invalid NULL pointer is specified for modularization!");
   }
   if (dynamic_cast<const G4PVDivision*>(physvol))
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "It is not possible to modularize by divisionvol!");
   }
   if (physvol->IsParameterised())
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "It is not possible to modularize by parameterised volume!");
   }
   if (physvol->IsReplicated())
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "It is not possible to modularize by replicated volume!");
   }

   PvolumeMap()[physvol] = fname;
}

void G4DAEWrite::AddModule(const G4int depth)
{
   if (depth<0)
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "Depth must be a positive number!");
   }
   if (DepthMap().find(depth) != DepthMap().end())
   {
     G4Exception("G4DAEWrite::AddModule()", "InvalidSetup", FatalException,
                 "Adding module(s) at this depth is already requested!");
   }
   DepthMap()[depth] = 0;
}

G4String G4DAEWrite::Modularize( const G4VPhysicalVolume* const physvol,
                                  const G4int depth )
{
   if (PvolumeMap().find(physvol) != PvolumeMap().end())
   {
     return PvolumeMap()[physvol]; // Modularize via physvol
   }

   if (DepthMap().find(depth) != DepthMap().end()) // Modularize via depth
   {
     std::stringstream stream;
     stream << "depth" << depth << "_module" << DepthMap()[depth] << ".gdml";
     DepthMap()[depth]++;           // There can be more modules at this depth!
     return G4String(stream.str());
   }

   return G4String(""); // Empty string for module name = no modularization
                        // was requested at that level/physvol!
}

void G4DAEWrite::SetAddPointerToName(G4bool set)
{
   addPointerToName = set;
}
