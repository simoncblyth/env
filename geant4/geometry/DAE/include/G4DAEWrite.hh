#ifndef _G4DAEWRITE_INCLUDED_
#define _G4DAEWRITE_INCLUDED_

#include <sys/stat.h>
#include <iostream>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include "G4LogicalVolume.hh"
#include "G4Transform3D.hh"
#include "G4PVDivision.hh"

class G4DAEWrite
{
   typedef std::map<const G4LogicalVolume*,G4Transform3D> VolumeMapType;
   typedef std::map<const G4VPhysicalVolume*,G4String> PhysVolumeMapType;
   typedef std::map<G4int,G4int> DepthMapType;

 public:  // without description

   G4Transform3D Write(const G4String& filename,
                       const G4LogicalVolume* const topLog,
                       const G4String& schemaPath,
                       const G4int depth, G4bool storeReferences=true);
   void AddModule(const G4VPhysicalVolume* const topVol);
   void AddModule(const G4int depth);
   static void SetAddPointerToName(G4bool);

 protected:

   G4String SchemaLocation;

   VolumeMapType& VolumeMap();

   G4String GenerateName(const G4String&,const void* const, G4bool ref=false );
   xercesc::DOMAttr* NewNCNameAttribute(const G4String&, const G4String&, G4bool ref=false);
   xercesc::DOMAttr* NewAttribute(const G4String&, const G4String&);
   xercesc::DOMAttr* NewAttribute(const G4String&, const G4double&);
   xercesc::DOMElement* NewElement(const G4String&);
   xercesc::DOMElement* NewElementOneAtt(const G4String& name, const G4String& att, const G4String& val);
   xercesc::DOMElement* NewElementOneNCNameAtt(const G4String& name, const G4String& att, const G4String& val, G4bool ref=false);
   xercesc::DOMElement* NewElementTwoAtt(const G4String& name, const G4String& att1, const G4String& val1, const G4String& att2, const G4String& val2);
   xercesc::DOMElement* NewTextElement(const G4String&, const G4String&);
   virtual void AssetWrite(xercesc::DOMElement*)=0;

   virtual void EffectsWrite(xercesc::DOMElement*)=0;
   virtual void MaterialsWrite(xercesc::DOMElement*)=0;
   virtual void SolidsWrite(xercesc::DOMElement*)=0;
   virtual void StructureWrite(xercesc::DOMElement*)=0;
   virtual G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const,
                                            const G4int)=0;
   virtual void SetupWrite(xercesc::DOMElement*,
                           const G4LogicalVolume* const)=0;
   G4String Modularize(const G4VPhysicalVolume* const topvol,
                       const G4int depth);

 private:

   G4bool FileExists(const G4String&) const;
   PhysVolumeMapType& PvolumeMap();
   DepthMapType& DepthMap();

 private:

   static G4bool addPointerToName;
   xercesc::DOMDocument* doc;
   //XMLCh tempStr[100];
   const static int tempStrSize = 256 ;
   XMLCh tempStr[tempStrSize];

};

#endif
