#include "G4DAEWriteStructure.hh"
#include "G4DAEPolyhedron.hh"    // for DAE WRL checking 
#include <sstream>

void
G4DAEWriteStructure::DivisionvolWrite(xercesc::DOMElement* volumeElement,
                                       const G4PVDivision* const divisionvol)
{
   EAxis axis = kUndefined;
   G4int number = 0;
   G4double width = 0.0;
   G4double offset = 0.0;
   G4bool consuming = false;

   divisionvol->GetReplicationData(axis,number,width,offset,consuming);

   G4String unitString("mm");
   G4String axisString("kUndefined");
   if (axis==kXAxis) { axisString = "kXAxis"; } else
   if (axis==kYAxis) { axisString = "kYAxis"; } else
   if (axis==kZAxis) { axisString = "kZAxis"; } else
   if (axis==kRho) { axisString = "kRho";     } else
   if (axis==kPhi) { axisString = "kPhi"; unitString = "degree"; }

   const G4String name
         = GenerateName(divisionvol->GetName(),divisionvol);
   const G4String volumeref
         = GenerateName(divisionvol->GetLogicalVolume()->GetName(),
                        divisionvol->GetLogicalVolume());

   xercesc::DOMElement* divisionvolElement = NewElement("divisionvol");
   divisionvolElement->setAttributeNode(NewAttribute("axis",axisString));
   divisionvolElement->setAttributeNode(NewAttribute("number",number));
   divisionvolElement->setAttributeNode(NewAttribute("width",width));
   divisionvolElement->setAttributeNode(NewAttribute("offset",offset));
   divisionvolElement->setAttributeNode(NewAttribute("unit",unitString));
   xercesc::DOMElement* volumerefElement = NewElement("volumeref");
   volumerefElement->setAttributeNode(NewAttribute("ref",volumeref));
   divisionvolElement->appendChild(volumerefElement);
   volumeElement->appendChild(divisionvolElement);
}

void G4DAEWriteStructure::MatrixWrite(xercesc::DOMElement* nodeElement, const G4Transform3D& T)
{
    std::ostringstream ss ;
    // row-major order 

    ss << "\n\t\t\t\t" ;
    ss << T.xx() << " " ;
    ss << T.xy() << " " ;
    ss << T.xz() << " " ;
    ss << T.dx() << "\n" ;

    ss << T.yx() << " " ;
    ss << T.yy() << " " ;
    ss << T.yz() << " " ;
    ss << T.dy() << "\n" ;

    ss << T.zx() << " " ;
    ss << T.zy() << " " ;
    ss << T.zz() << " " ;
    ss << T.dz() << "\n" ;

    ss << "0.0 0.0 0.0 1.0\n" ;

    std::string fourbyfour = ss.str(); 
    xercesc::DOMElement* matrixElement = NewTextElement("matrix", fourbyfour);
    nodeElement->appendChild(matrixElement);
}


void G4DAEWriteStructure::PhysvolWrite(xercesc::DOMElement* parentNodeElement,
                                        const G4VPhysicalVolume* const physvol,
                                        const G4Transform3D& T,
                                        const G4String& ModuleName)
{

   // DEBUG FOR WRL-DAE CORRESPONDENCE
   // NO GOOD FOR WRL COMPARISON, AS NEED TO TRAVERSE THE NODE TREE TO VISIT THEM ALL

   /*
   std::string polysmry ; 
   {
       G4bool recPoly = GetRecreatePoly(); 
       G4DAEPolyhedron poly(physvol->GetLogicalVolume()->GetSolid(), recPoly );  // recPoly always creates a new poly, even when one exists already   
       std::stringstream ss ; 
       ss << "n " << physvol->GetName() << "." << physvol->GetCopyNo() << " " ; 
       ss << "v " << poly.GetNoVertices() << " " ; 
       ss << "f " << poly.GetNoFacets() << " " ; 
       polysmry = ss.str();
   }

   fSummary.push_back(polysmry); 
   */

   const G4String pvname = GenerateName(physvol->GetName(),physvol);
   const G4String lvname = GenerateName(physvol->GetLogicalVolume()->GetName(),physvol->GetLogicalVolume() );

   G4int copyNo = physvol->GetCopyNo();  
   xercesc::DOMElement* childNodeElement = NewElementOneNCNameAtt("node","id",pvname);
  /*
   //
   // NODE RESUSE MEANS CANNOT ASSIGN A USEFUL INDEX AT THIS STAGE
   // THE INDEX ONLY "HAPPENS" ONCE YOU FLATTEN THE TREE BY TRAVERSAL
   //
   G4int index = ++fNodeIndex ; 
   std::string nis ;
   {
       std::ostringstream ss ;
       ss << index ; 
       nis = ss.str();
   }
   childNodeElement->setAttributeNode(NewAttribute("name",nis));
  */

   MatrixWrite( childNodeElement, T );

   xercesc::DOMElement* instanceNodeElement = NewElementOneNCNameAtt("instance_node", "url", lvname , true);
   childNodeElement->appendChild(instanceNodeElement);

   // extra/meta
   xercesc::DOMElement* extraElement = NewElement("extra");

   xercesc::DOMElement* metaElement = NewElementOneAtt("meta", "id", pvname);
   std::ostringstream ss ;
   ss << copyNo ; 
   metaElement->appendChild(NewTextElement("copyNo",ss.str()));
   metaElement->appendChild(NewTextElement("ModuleName",ModuleName));
   //metaElement->appendChild(NewTextElement("polysmry",polysmry));
   extraElement->appendChild(metaElement);

   childNodeElement->appendChild(extraElement);
   parentNodeElement->appendChild(childNodeElement);
}

void G4DAEWriteStructure::ReplicavolWrite(xercesc::DOMElement* volumeElement,
                                     const G4VPhysicalVolume* const replicavol)
{
   EAxis axis = kUndefined;
   G4int number = 0;
   G4double width = 0.0;
   G4double offset = 0.0;
   G4bool consuming = false;
   G4String unitString("mm");

   replicavol->GetReplicationData(axis,number,width,offset,consuming);

   const G4String volumeref
         = GenerateName(replicavol->GetLogicalVolume()->GetName(),
                        replicavol->GetLogicalVolume());

   xercesc::DOMElement* replicavolElement = NewElement("replicavol");
   replicavolElement->setAttributeNode(NewAttribute("number",number));
   xercesc::DOMElement* volumerefElement = NewElement("volumeref");
   volumerefElement->setAttributeNode(NewAttribute("ref",volumeref));
   replicavolElement->appendChild(volumerefElement);

   xercesc::DOMElement* replicateElement = NewElement("replicate_along_axis");
   replicavolElement->appendChild(replicateElement);

   xercesc::DOMElement* dirElement = NewElement("direction");
   if(axis==kXAxis)dirElement->setAttributeNode(NewAttribute("x","1"));
   if(axis==kYAxis)dirElement->setAttributeNode(NewAttribute("y","1"));
   if(axis==kZAxis)dirElement->setAttributeNode(NewAttribute("z","1"));
   if(axis==kRho)dirElement->setAttributeNode(NewAttribute("rho","1"));
   if(axis==kPhi)dirElement->setAttributeNode(NewAttribute("phi","1"));
   replicateElement->appendChild(dirElement);

   xercesc::DOMElement* widthElement = NewElement("width");
   widthElement->setAttributeNode(NewAttribute("value",width));
   widthElement->setAttributeNode(NewAttribute("unit",unitString));
   replicateElement->appendChild(widthElement);

   xercesc::DOMElement* offsetElement = NewElement("offset");
   offsetElement->setAttributeNode(NewAttribute("value",offset));
   offsetElement->setAttributeNode(NewAttribute("unit",unitString));
   replicateElement->appendChild(offsetElement);

   volumeElement->appendChild(replicavolElement);
}

void G4DAEWriteStructure::StructureWrite(xercesc::DOMElement* daeElement)
{
   G4cout << "G4DAE: Writing structure/library_nodes..." << G4endl;

   structureElement = NewElement("library_nodes");
   daeElement->appendChild(structureElement);
}

/*
void G4DAEWriteStructure::SetVisAttributes (const G4VisAttributes& VA)
{
   fVisAttributes = new G4VisAttributes(VA);
}
*/

G4Transform3D G4DAEWriteStructure::
TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
{
   // "NEAR" GEOMETRY PASSES HERE 5642 TIME ONLY AS  THIS IS LV (NOT PV) 
   // FOR THE FULL 12230 SEE PhysvolWrite  

   if (VolumeMap().find(volumePtr) != VolumeMap().end())
   {
       return VolumeMap()[volumePtr]; // Volume is already processed
   }

   //
   // Compiler takes exception to::
   //
   //    volumePtr->SetVisAttributes(fVisAttributes);   
   //
   // due to const correctness from this methods signature 
   // preventing setting the VisAttributes on the volume (at compile time)
   // so need to attack the polyhedron, as done by::
   //
   //      void G4VSceneHandler::RequestPrimitives (const G4VSolid& solid) 
   //
   // from visualization/management/src/G4VSceneHandler.cc
   //
   //

   G4VSolid* solidPtr = volumePtr->GetSolid();
   G4Transform3D R,invR;

   const G4String lvname = GenerateName(volumePtr->GetName(),volumePtr);

   G4Material* materialPtr = volumePtr->GetMaterial();
   G4String matSymbol = GenerateMaterialSymbol(materialPtr->GetName()) ;  

   const G4String matname = GenerateName(materialPtr->GetName(), materialPtr );
   const G4String geoname = GenerateName(solidPtr->GetName(), solidPtr );

   G4bool ref = true ; 
   xercesc::DOMElement* nodeElement = NewElementOneNCNameAtt("node","id", lvname);
   xercesc::DOMElement* igElement = NewElementOneNCNameAtt("instance_geometry","url", geoname, ref);
   xercesc::DOMElement* bmElement = NewElement("bind_material");
   xercesc::DOMElement* tcElement = NewElement("technique_common");
   xercesc::DOMElement* imElement = NewElementOneNCNameAtt("instance_material", "target", matname, ref );
   imElement->setAttributeNode(NewAttribute("symbol", matSymbol ));
   tcElement->appendChild(imElement);
   bmElement->appendChild(tcElement);
   igElement->appendChild(bmElement);
   nodeElement->appendChild(igElement);

   const G4int daughterCount = volumePtr->GetNoDaughters();


   // NB the heirarchy is divied out into multiple nodes

   for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
   {
      const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
      const G4String ModuleName = Modularize(physvol,depth);

      G4Transform3D daughterR;

      daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);

      G4RotationMatrix rot, invrot;
      if (physvol->GetFrameRotation() != 0)
      {
         rot = *(physvol->GetFrameRotation());
         invrot = rot.inverse();
      }

      // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
      G4Transform3D P(invrot,physvol->GetObjectTranslation());

      PhysvolWrite(nodeElement,physvol,invR*P*daughterR,ModuleName);
   }


   structureElement->appendChild(nodeElement);

   // Append the volume AFTER traversing the children so that
   // the order of volumes will be correct!

   VolumeMap()[volumePtr] = R;

   G4DAEWriteEffects::AddEffectMaterial(materialPtr);
   G4DAEWriteMaterials::AddMaterial(materialPtr);
   G4DAEWriteSolids::AddSolid(solidPtr, matSymbol);

   return R;
}
