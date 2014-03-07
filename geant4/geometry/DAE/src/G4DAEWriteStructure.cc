#include "G4DAEWriteStructure.hh"
#include "G4DAEPolyhedron.hh"    // for DAE WRL checking 
#include <sstream>

#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"


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



/*
 * Creates "bordersurface" element from G4LogicalBorderSurface* with physvolref contained
 * elements
 *
 *     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop">   
 *          <physvolref ref="/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap0xb6d81c8"/>  
 *          <physvolref ref="/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR0xb657cf0"/>
 *     </bordersurface>
 *
 */

void G4DAEWriteStructure::
BorderSurfaceCache(const G4LogicalBorderSurface* const bsurf)
{
   if (!bsurf)  { return; }

   const G4SurfaceProperty* psurf = bsurf->GetSurfaceProperty();

   // Generate the new element for border-surface
   //
   xercesc::DOMElement* borderElement = NewElement("bordersurface");
   borderElement->setAttributeNode(NewNCNameAttribute("name", bsurf->GetName()));
   borderElement->setAttributeNode(NewNCNameAttribute("surfaceproperty", psurf->GetName()));

   const G4String volumeref1 = GenerateName(bsurf->GetVolume1()->GetName(),
                                            bsurf->GetVolume1());
   const G4String volumeref2 = GenerateName(bsurf->GetVolume2()->GetName(),
                                            bsurf->GetVolume2());
   xercesc::DOMElement* volumerefElement1 = NewElement("physvolref");
   xercesc::DOMElement* volumerefElement2 = NewElement("physvolref");
   volumerefElement1->setAttributeNode(NewNCNameAttribute("ref",volumeref1));
   volumerefElement2->setAttributeNode(NewNCNameAttribute("ref",volumeref2));
   borderElement->appendChild(volumerefElement1);
   borderElement->appendChild(volumerefElement2);

   if (FindOpticalSurface(psurf))
   {
     const G4OpticalSurface* opsurf =
       dynamic_cast<const G4OpticalSurface*>(psurf);
     if (!opsurf)
     {
       G4Exception("G4DAEWriteStructure::BorderSurfaceCache()",
                   "InvalidSetup", FatalException, "No optical surface found!");
       return;
     }
     OpticalSurfaceWrite(extrasurfElement, opsurf);
   }

   borderElementVec.push_back(borderElement);
}



/*
 * Create opticalsurface element with attributes from G4OpticalSurface*
 * append to first argument element
 * 
 * from G4GDMLWriteSolids::OpticalSurfaceWrite
 */
void G4DAEWriteStructure::
OpticalSurfaceWrite(xercesc::DOMElement* targetElement,
                    const G4OpticalSurface* const surf)
{
   xercesc::DOMElement* optElement = NewElement("opticalsurface");
   G4OpticalSurfaceModel smodel = surf->GetModel();
   G4double sval = (smodel==glisur) ? surf->GetPolish() : surf->GetSigmaAlpha();

   optElement->setAttributeNode(NewNCNameAttribute("name", surf->GetName()));
   optElement->setAttributeNode(NewAttribute("model", smodel));
   optElement->setAttributeNode(NewAttribute("finish", surf->GetFinish()));
   optElement->setAttributeNode(NewAttribute("type", surf->GetType()));
   optElement->setAttributeNode(NewAttribute("value", sval));

   G4MaterialPropertiesTable* ptable = surf->GetMaterialPropertiesTable();
   PropertyWrite( optElement, ptable );

   targetElement->appendChild(optElement);
}


/*
 *  Create skinsurface element for the G4LogicalSkinSurface* with volumeref contained
 *  record the G4OpticalSurface* in the opt_vec member
 *  add skinsurface element to the skinElementVec member 
 */

void G4DAEWriteStructure::
SkinSurfaceCache(const G4LogicalSkinSurface* const ssurf)
{
   if (!ssurf)  { return; }

   const G4SurfaceProperty* psurf = ssurf->GetSurfaceProperty();

   // Generate the new element for border-surface
   //
   xercesc::DOMElement* skinElement = NewElement("skinsurface");
   skinElement->setAttributeNode(NewNCNameAttribute("name", ssurf->GetName()));
   skinElement->setAttributeNode(NewNCNameAttribute("surfaceproperty",psurf->GetName()));
   const G4String volumeref = GenerateName(ssurf->GetLogicalVolume()->GetName(), ssurf->GetLogicalVolume());
   xercesc::DOMElement* volumerefElement = NewElement("volumeref");
   volumerefElement->setAttributeNode(NewNCNameAttribute("ref",volumeref));
   skinElement->appendChild(volumerefElement);

   if (FindOpticalSurface(psurf))
   {
     const G4OpticalSurface* opsurf =
       dynamic_cast<const G4OpticalSurface*>(psurf);
     if (!opsurf)
     {
       G4Exception("G4DAEWriteStructure::SkinSurfaceCache()",
                   "InvalidSetup", FatalException, "No optical surface found!");
       return;
     }
     OpticalSurfaceWrite(extrasurfElement, opsurf);
   }

   skinElementVec.push_back(skinElement);
}


/*
 *  Cast G4SurfaceProperty* to G4OpticalSurface* and append to opt_vec member if not already there
 *  returns 
 *       true, if not already present
 *       false, if already collected 
 * 
 */
G4bool G4DAEWriteStructure::FindOpticalSurface(const G4SurfaceProperty* psurf)
{
   const G4OpticalSurface* osurf = dynamic_cast<const G4OpticalSurface*>(psurf);
   std::vector<const G4OpticalSurface*>::const_iterator pos;
   pos = std::find(opt_vec.begin(), opt_vec.end(), osurf);
   if (pos != opt_vec.end()) { return false; }  // item already created!

   opt_vec.push_back(osurf);              // cache it for future reference
   return true;
}


/*
 *  Return first G4LogicalSkinSurface* associated with a G4LogicalVolume* 
 */
const G4LogicalSkinSurface*
G4DAEWriteStructure::GetSkinSurface(const G4LogicalVolume* const lvol)
{
  G4LogicalSkinSurface* surf = 0;
  G4int nsurf = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();
  if (nsurf)
  {
    const G4LogicalSkinSurfaceTable* stable =
          G4LogicalSkinSurface::GetSurfaceTable();
    std::vector<G4LogicalSkinSurface*>::const_iterator pos;
    for (pos = stable->begin(); pos != stable->end(); pos++)
    {
      if (lvol == (*pos)->GetLogicalVolume())
      {
        surf = *pos; break;
      }
    }
  }
  return surf;
}


/*
 *  Return first G4LogicalBorderSurface* associated with a G4VPhysicalVolume* 
 *  hmm no regard for CopyNo ?
 */
const G4LogicalBorderSurface*
G4DAEWriteStructure::GetBorderSurface(const G4VPhysicalVolume* const pvol)
{

  // hmm should the parent be checked ? 

  G4LogicalBorderSurface* surf_first_pv1 = 0;
  G4int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
  if (nsurf)
  {
    G4cout << "G4DAE::GetBorderSurface ... " << pvol->GetName() << "[" << pvol->GetCopyNo() << "]" << G4endl;
    const G4LogicalBorderSurfaceTable* btable =
          G4LogicalBorderSurface::GetSurfaceTable();
    std::vector<G4LogicalBorderSurface*>::const_iterator pos;
    for (pos = btable->begin(); pos != btable->end(); pos++)
    {
      const G4VPhysicalVolume* pv1 = (*pos)->GetVolume1() ;
      const G4VPhysicalVolume* pv2 = (*pos)->GetVolume2() ;
 
      if (pvol == pv1)  
      {                                  
           if( surf_first_pv1 == 0)
           {
                surf_first_pv1 = *pos ;
                G4cout << "G4DAE::GetBorderSurface surf_first_pv1 "<< G4endl ;
                G4cout << "         PV1 [copyNo]name [" << pv1->GetCopyNo() << "]" << pv1->GetName() << G4endl;
                G4cout << "         PV2 [copyNo]name [" << pv2->GetCopyNo() << "]" << pv2->GetName() << G4endl;
           }
           else
           {
                G4cout << "G4DAE::GetBorderSurface surf other PV1 match "<< G4endl ;
                G4cout << "         PV1 [copyNo]name [" << pv1->GetCopyNo() << "]" << pv1->GetName() << G4endl;
                G4cout << "         PV2 [copyNo]name [" << pv2->GetCopyNo() << "]" << pv2->GetName() << G4endl;
           }

           //surf = *pos; break;

      }  // pv1 match


      if (pvol == pv2)
      {
            G4cout << "G4DAE::GetBorderSurface surf PV2 match "<< G4endl ;
            G4cout << "         PV1 [copyNo]name [" << pv1->GetCopyNo() << "]" << pv1->GetName() << G4endl;
            G4cout << "         PV2 [copyNo]name [" << pv2->GetCopyNo() << "]" << pv2->GetName() << G4endl;
 
      }  // pv2 match 

    }   // over surfaces
  }   // non zero border surfaces

  return surf_first_pv1 ;  // keep same function as GDML for now
}


xercesc::DOMElement* G4DAEWriteStructure::GetPVElement(const G4VPhysicalVolume* const pv)
{
    xercesc::DOMElement* pvElement = NewElement("pv");
    const G4String volumeref = GenerateName(pv->GetName(),pv);
    pvElement->setAttributeNode(NewNCNameAttribute("ref",volumeref));
    pvElement->setAttributeNode(NewNCNameAttribute("name",pv->GetName()));
    pvElement->setAttributeNode(NewAttribute("copyNo",pv->GetCopyNo()));
    return pvElement ;  
}


xercesc::DOMElement* G4DAEWriteStructure::GetBorderSurfacesMetaElement()
{
    xercesc::DOMElement* metaElement = NewElement("meta");
    const G4VPhysicalVolume* pv1 ;
    const G4VPhysicalVolume* pv2 ;
    const G4SurfaceProperty* psurf ; 
    const G4LogicalBorderSurfaceTable* btable = G4LogicalBorderSurface::GetSurfaceTable();
    std::vector<G4LogicalBorderSurface*>::const_iterator bsurf;
    for (bsurf = btable->begin(); bsurf != btable->end(); bsurf++)
    {
        pv1 = (*bsurf)->GetVolume1() ;
        pv2 = (*bsurf)->GetVolume2() ;

        psurf = (*bsurf)->GetSurfaceProperty();

        xercesc::DOMElement* bsurfElement = NewElement("bsurf");
        bsurfElement->setAttributeNode(NewNCNameAttribute("name", (*bsurf)->GetName()));
        bsurfElement->setAttributeNode(NewNCNameAttribute("surfaceproperty", psurf->GetName()));

        xercesc::DOMElement* pv1Elem = GetPVElement( pv1 ); 
        xercesc::DOMElement* pv2Elem = GetPVElement( pv2 ); 
        bsurfElement->appendChild(pv1Elem);
        bsurfElement->appendChild(pv2Elem);
        metaElement->appendChild(bsurfElement);
    }
    return metaElement ; 
}

void G4DAEWriteStructure::SurfacesWrite()
{
   G4cout << "G4DAE: Writing surfaces..." << G4endl;

   std::vector<xercesc::DOMElement*>::const_iterator pos;
   for (pos = skinElementVec.begin(); pos != skinElementVec.end(); pos++)
   {
     extrasurfElement->appendChild(*pos);
   }
   for (pos = borderElementVec.begin(); pos != borderElementVec.end(); pos++)
   {
     extrasurfElement->appendChild(*pos);
   }

   // debugging ambiguous PV issue
   xercesc::DOMElement* metaElement = GetBorderSurfacesMetaElement();
   extrasurfElement->appendChild(metaElement);

   //
   // place here after Traverse as COLLADA spec 
   // requires "extra" child elements of "library_nodes"
   // to be after all the "node"
   // 
   structureElement->appendChild(extrasurfElement); // 
}






void G4DAEWriteStructure::StructureWrite(xercesc::DOMElement* daeElement)
{
   G4cout << "G4DAE: Writing structure/library_nodes..." << G4endl;

   structureElement = NewElement("library_nodes");
   extrasurfElement = NewElement("extra");

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
      BorderSurfaceCache(GetBorderSurface(physvol));
   }


   structureElement->appendChild(nodeElement);

   // Append the volume AFTER traversing the children so that
   // the order of volumes will be correct!

   VolumeMap()[volumePtr] = R;

   G4DAEWriteEffects::AddEffectMaterial(materialPtr);
   G4DAEWriteMaterials::AddMaterial(materialPtr);
   G4DAEWriteSolids::AddSolid(solidPtr, matSymbol);

   SkinSurfaceCache(GetSkinSurface(volumePtr));

   return R;
}
