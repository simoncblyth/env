#include "G4DAEWrite.hh"
#include <algorithm>  
#include <xercesc/util/XMLChar.hpp>
#include "G4DAEUtil.hh"

#include "G4Material.hh"
#include "G4MaterialPropertyVector.hh"




G4bool G4DAEWrite::addPointerToName = true;
G4bool G4DAEWrite::recreatePoly = false ;

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

G4String G4DAEWrite::GenerateMaterialSymbol(const G4String& mat)
{
   G4String matSym ; 
   G4String prefix("/dd/Materials/");
   if (!mat.compare(0, prefix.size(), prefix)){
       matSym = mat.substr(prefix.size());
   } else {
       matSym = mat ; 
   } 
   return matSym ; 
}

G4String G4DAEWrite::GenerateTexturePath(const G4String& name )
{
   std::stringstream ss;
   ss << "./textures/" ; 
   ss << name ; 
   ss << ".png" ;

   G4String texpath(ss.str());
   return texpath ;  
}

G4String G4DAEWrite::GenerateName(const G4String& name, const void* const ptr, G4bool ref )
{
   G4String nameOut;
   std::stringstream stream; 
   if(ref) stream << "#" ; 
   stream << name;
   if (addPointerToName) { stream << ptr; };

   nameOut=G4String(stream.str());

   if(nameOut.contains(' '))
   nameOut.erase(std::remove(nameOut.begin(),nameOut.end(),' '),nameOut.end());
   return nameOut;
}

xercesc::DOMAttr* G4DAEWrite::NewNCNameAttribute(const G4String& name,
                                            const G4String& value, G4bool ref)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   xercesc::DOMAttr* att = doc->createAttribute(tempStr);

   G4String val(value);
   G4DAEUtil::encodeNCName( val );

   int offset = 0 ;
   if(ref){
       G4String hash("#");
       val = hash + val ; 
       offset = 1; 
   }
   
   xercesc::XMLString::transcode(val,tempStr,tempStrSize-1);

   // if(!xercesc::XMLString::isValidNCName(tempStr+offset))
   // XMLString::isValidNCName is deprecated
   if(!xercesc::XMLChar1_0::isValidNCName(tempStr+offset, 
                                          xercesc::XMLString::stringLen(tempStr)-offset)){
      G4cout << "WARNING despite encoding still not a valid NCName " <<  name << " " << val << G4endl; 
   }

   att->setValue(tempStr);
   return att;
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

xercesc::DOMElement* G4DAEWrite::NewElementOneAtt(const G4String& name, const G4String& att, const G4String& val)
{
   xercesc::DOMElement* element = NewElement(name);
   element->setAttributeNode(NewAttribute(att,val));
   return element ; 
}

xercesc::DOMElement* G4DAEWrite::NewElementOneNCNameAtt(const G4String& name, const G4String& att, const G4String& val, G4bool ref )
{
   xercesc::DOMElement* element = NewElement(name);
   element->setAttributeNode(NewNCNameAttribute(att,val,ref));
   return element ; 
}


xercesc::DOMElement* G4DAEWrite::NewElementTwoAtt(const G4String& name, const G4String& att1, const G4String& val1, const G4String& att2, const G4String& val2)
{
   xercesc::DOMElement* element = NewElement(name);
   element->setAttributeNode(NewAttribute(att1,val1));
   element->setAttributeNode(NewAttribute(att2,val2));
   return element ; 
}



xercesc::DOMElement* G4DAEWrite::NewTextElement(const G4String& name, const G4String& text)
{
   xercesc::XMLString::transcode(name,tempStr,tempStrSize-1);
   xercesc::DOMElement* e = doc->createElement(tempStr);

   // text content can potentially be much larger than tags/attributes 
   // so transcode dynamically rather than with a fixed buffer
   XMLCh* content = xercesc::XMLString::transcode(text);
   e->setTextContent(content);
   xercesc::XMLString::release(&content);

   return e; 
}




G4Transform3D G4DAEWrite::Write(const G4String& fname,
                                 const G4LogicalVolume* const logvol,
                                 const G4String& setSchemaLocation,
                                 const G4int depth,
                                       G4bool refs,
                                       G4bool _recreatePoly,
                                       G4int nodeIndex )
{
   SchemaLocation = setSchemaLocation;
   addPointerToName = refs;
   recreatePoly = _recreatePoly ; 
   fNodeIndex = nodeIndex ; 

   G4cout << "G4DAEWrite::Write addPointerToName " << addPointerToName << " recreatePoly " << recreatePoly << " nodeindex " << fNodeIndex << G4endl ;  
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
   SolidsWrite(dae);   // geometry before materials to match pycollada
   MaterialsWrite(dae);

   StructureWrite(dae);   // writing order does not follow inheritance order

   SetupWrite(dae, logvol);

   G4Transform3D R = TraverseVolumeTree(logvol,depth);

   SurfacesWrite(); 

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


   //fSummary.push_back(std::string("klop"));

   G4String smry(fname);
   smry += ".txt" ; 
   G4cout << "G4DAEWrite writing summary lines '" << smry << G4endl;
   G4DAEUtil::WriteLines( smry, fSummary );

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

void G4DAEWrite::SetRecreatePoly(G4bool set)
{
   recreatePoly = set;
}

G4bool G4DAEWrite::GetRecreatePoly()
{
   return recreatePoly ;
}



// adapted from /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLWriteMaterials.cc
// needs access to property map, so must patch older geant4 to have access to the map
//
// relies on API from the G4MaterialPropertyVector 
// "spill the beans" patch for older geant4.9.2.p01
// patch should not be needed in newer geant4 that typedefs
// G4MaterialPropertyVector from G4PhysicsOrderedFreeVector  
//
//

void G4DAEWrite::PropertyVectorWrite(const G4String& key,
                           const G4MaterialPropertyVector* const pvec, 
                            xercesc::DOMElement* extraElement)
{

   std::ostringstream pvalues;

#ifdef _GEANT4_TMP_GEANT94_
   for (G4int i=0; i<pvec->Entries(); i++)                                      
   {                                                                            
     G4MPVEntry cval = pvec->GetEntry(i);                                       
     if (i!=0)  { pvalues << " "; }                                             
     pvalues << cval.GetPhotonEnergy() << " " << cval.GetProperty();            
   }        
#else
   for (size_t i=0; i<pvec->GetVectorLength(); i++)
   {
       if (i!=0)  { pvalues << " "; }
       pvalues << pvec->Energy(i) << " " << (*pvec)[i];
   }
#endif



   xercesc::DOMElement* matrixElement = NewTextElement("matrix",pvalues.str());
   const G4String matrixref = GenerateName(key, pvec);
   matrixElement->setAttributeNode(NewAttribute("name", matrixref));
   matrixElement->setAttributeNode(NewAttribute("coldim", "2"));

   extraElement->appendChild(matrixElement);  // was toplevel defineElement for GDML
}



void G4DAEWrite::PropertyWrite(xercesc::DOMElement* extraElement,  const G4MaterialPropertiesTable* const ptable)
{
   xercesc::DOMElement* propElement;
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
         G4String warn_message = "Null pointer for material property -" + mpos->first ;
         G4Exception("G4DAEWrite::PropertyWrite()", "NullPointer",
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





