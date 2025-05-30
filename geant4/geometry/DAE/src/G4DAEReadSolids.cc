//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: G4DAEReadSolids.cc,v 1.22 2008/11/21 09:32:46 gcosmo Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
// class G4DAEReadSolids Implementation
//
// Original author: Zoltan Torzsok, November 2007
//
// --------------------------------------------------------------------

#include "G4DAEReadSolids.hh"

void G4DAEReadSolids::
BooleanRead(const xercesc::DOMElement* const booleanElement, const BooleanOp op)
{
   G4String name;
   G4String first;
   G4String second;
   G4ThreeVector position(0.0,0.0,0.0);
   G4ThreeVector rotation(0.0,0.0,0.0);
   G4ThreeVector firstposition(0.0,0.0,0.0);
   G4ThreeVector firstrotation(0.0,0.0,0.0);

   const xercesc::DOMNamedNodeMap* const attributes
         = booleanElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name")  { name = GenerateName(attValue); }
   }

   for (xercesc::DOMNode* iter = booleanElement->getFirstChild();
        iter != 0;iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="first") { first = RefRead(child); } else
      if (tag=="second") { second = RefRead(child); } else
      if (tag=="position") { VectorRead(child,position); } else
      if (tag=="rotation") { VectorRead(child,rotation); } else
      if (tag=="positionref")
        { position = GetPosition(GenerateName(RefRead(child))); } else
      if (tag=="rotationref")
        { rotation = GetRotation(GenerateName(RefRead(child))); } else
      if (tag=="firstposition") { VectorRead(child,firstposition); } else
      if (tag=="firstrotation") { VectorRead(child,firstrotation); } else
      if (tag=="firstpositionref")
        { firstposition = GetPosition(GenerateName(RefRead(child))); } else
      if (tag=="firstrotationref")
        { firstrotation = GetRotation(GenerateName(RefRead(child))); } 
      else
      {
        G4String error_msg = "Unknown tag in boolean solid: " + tag;
        G4Exception("G4DAEReadSolids::BooleanRead()", "ReadError",
                    FatalException, error_msg);
      }
   }

   G4VSolid* firstSolid = GetSolid(GenerateName(first));
   G4VSolid* secondSolid = GetSolid(GenerateName(second));

   G4Transform3D transform(GetRotationMatrix(rotation),position);

   if (( (firstrotation.x()!=0.0) || (firstrotation.y()!=0.0)
                                  || (firstrotation.z()!=0.0))
    || ( (firstposition.x()!=0.0) || (firstposition.y()!=0.0)
                                  || (firstposition.z()!=0.0)))
   { 
      G4Transform3D firsttransform(GetRotationMatrix(firstrotation),
                                   firstposition);
      firstSolid = new G4DisplacedSolid(GenerateName("displaced_"+first),
                                        firstSolid, firsttransform);
   }

   if (op==UNION)
     { new G4UnionSolid(name,firstSolid,secondSolid,transform); } else
   if (op==SUBTRACTION)
     { new G4SubtractionSolid(name,firstSolid,secondSolid,transform); } else
   if (op==INTERSECTION)
     { new G4IntersectionSolid(name,firstSolid,secondSolid,transform); }
}

void G4DAEReadSolids::BoxRead(const xercesc::DOMElement* const boxElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double x = 0.0;
   G4double y = 0.0;
   G4double z = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = boxElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="x") { x = eval.Evaluate(attValue); } else
      if (attName=="y") { y = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); }
   }

   x *= 0.5*lunit;
   y *= 0.5*lunit;
   z *= 0.5*lunit;

   new G4Box(name,x,y,z);
}

void G4DAEReadSolids::ConeRead(const xercesc::DOMElement* const coneElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rmin1 = 0.0;
   G4double rmax1 = 0.0;
   G4double rmin2 = 0.0;
   G4double rmax2 = 0.0;
   G4double z = 0.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = coneElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rmin1") { rmin1 = eval.Evaluate(attValue); } else
      if (attName=="rmax1") { rmax1 = eval.Evaluate(attValue); } else
      if (attName=="rmin2") { rmin2 = eval.Evaluate(attValue); } else
      if (attName=="rmax2") { rmax2 = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
   }

   rmin1 *= lunit;
   rmax1 *= lunit;
   rmin2 *= lunit;
   rmax2 *= lunit;
   z *= 0.5*lunit;
   startphi *= aunit;
   deltaphi *= aunit;

   new G4Cons(name,rmin1,rmax1,rmin2,rmax2,z,startphi,deltaphi);
}

void G4DAEReadSolids::
ElconeRead(const xercesc::DOMElement* const elconeElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double dx = 0.0;
   G4double dy = 0.0;
   G4double zmax = 0.0;
   G4double zcut = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = elconeElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="dx") { dx = eval.Evaluate(attValue); } else
      if (attName=="dy") { dy = eval.Evaluate(attValue); } else
      if (attName=="zmax") { zmax = eval.Evaluate(attValue); } else
      if (attName=="zcut") { zcut = eval.Evaluate(attValue); }
   }

   dx *= lunit;
   dy *= lunit;
   zmax *= lunit;
   zcut *= lunit;

   new G4EllipticalCone(name,dx,dy,zmax,zcut);
}

void G4DAEReadSolids::
EllipsoidRead(const xercesc::DOMElement* const ellipsoidElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double ax = 0.0;
   G4double by = 0.0;
   G4double cz = 0.0;
   G4double zcut1 = 0.0;
   G4double zcut2 = 0.0; 

   const xercesc::DOMNamedNodeMap* const attributes
         = ellipsoidElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name  = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="ax") { ax = eval.Evaluate(attValue); } else
      if (attName=="by") { by = eval.Evaluate(attValue); } else
      if (attName=="cz") { cz = eval.Evaluate(attValue); } else
      if (attName=="zcut1") { zcut1 = eval.Evaluate(attValue); } else
      if (attName=="zcut2") { zcut2 = eval.Evaluate(attValue); }
   }

   ax *= lunit;
   by *= lunit;
   cz *= lunit;
   zcut1 *= lunit;
   zcut2 *= lunit; 

   new G4Ellipsoid(name,ax,by,cz,zcut1,zcut2);
}

void G4DAEReadSolids::
EltubeRead(const xercesc::DOMElement* const eltubeElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double dx = 0.0;
   G4double dy = 0.0;
   G4double dz = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = eltubeElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="dx") { dx = eval.Evaluate(attValue); } else
      if (attName=="dy") { dy = eval.Evaluate(attValue); } else
      if (attName=="dz") { dz = eval.Evaluate(attValue); }
   }

   dx *= lunit;
   dy *= lunit;
   dz *= lunit;

   new G4EllipticalTube(name,dx,dy,dz);
}

void G4DAEReadSolids::XtruRead(const xercesc::DOMElement* const xtruElement)
{
   G4String name;
   G4double lunit = 1.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = xtruElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; } 

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); }
   }

   std::vector<G4TwoVector> twoDimVertexList;
   std::vector<G4ExtrudedSolid::ZSection> sectionList;

   for (xercesc::DOMNode* iter = xtruElement->getFirstChild();
        iter != 0; iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE) { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="twoDimVertex")
        { twoDimVertexList.push_back(TwoDimVertexRead(child,lunit)); } else
      if (tag=="section")
        { sectionList.push_back(SectionRead(child,lunit)); }
   }

   new G4ExtrudedSolid(name,twoDimVertexList,sectionList);
}

void G4DAEReadSolids::HypeRead(const xercesc::DOMElement* const hypeElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rmin = 0.0;
   G4double rmax = 0.0;
   G4double inst = 0.0;
   G4double outst = 0.0;
   G4double z = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = hypeElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rmin") { rmin = eval.Evaluate(attValue); } else
      if (attName=="rmax") { rmax = eval.Evaluate(attValue); } else
      if (attName=="inst") { inst = eval.Evaluate(attValue); } else
      if (attName=="outst") { outst = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); }
   }

   rmin *= lunit;
   rmax *= lunit;
   inst *= aunit;
   outst *= aunit;
   z *= 0.5*lunit;

   new G4Hype(name,rmin,rmax,inst,outst,z);
}

void G4DAEReadSolids::OrbRead(const xercesc::DOMElement* const orbElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double r = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = orbElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="r") { r = eval.Evaluate(attValue); }
   }

   r *= lunit;

   new G4Orb(name,r);
}

void G4DAEReadSolids::ParaRead(const xercesc::DOMElement* const paraElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double x = 0.0;
   G4double y = 0.0;
   G4double z = 0.0;
   G4double alpha = 0.0;
   G4double theta = 0.0;
   G4double phi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = paraElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="x") { x = eval.Evaluate(attValue); } else
      if (attName=="y") { y = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="alpha") { alpha = eval.Evaluate(attValue); } else
      if (attName=="theta") { theta = eval.Evaluate(attValue); } else
      if (attName=="phi") { phi = eval.Evaluate(attValue); }
   }

   x *= 0.5*lunit;
   y *= 0.5*lunit;
   z *= 0.5*lunit;
   alpha *= aunit;
   theta *= aunit;
   phi *= aunit;

   new G4Para(name,x,y,z,alpha,theta,phi);
}

void G4DAEReadSolids::
ParaboloidRead(const xercesc::DOMElement* const paraElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rlo = 0.0;
   G4double rhi = 0.0;
   G4double dz = 0.0;
  
   const xercesc::DOMNamedNodeMap* const attributes
         = paraElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name")  { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rlo")   { rlo =  eval.Evaluate(attValue); } else
      if (attName=="rhi")   { rhi = eval.Evaluate(attValue); } else
      if (attName=="dz")    { dz = eval.Evaluate(attValue); } 
   }     

   rlo *= 1.*lunit;
   rhi *= 1.*lunit;
   dz *= 1.*lunit;
  
   new G4Paraboloid(name,dz,rlo,rhi);
}

void G4DAEReadSolids::
PolyconeRead(const xercesc::DOMElement* const polyconeElement) 
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = polyconeElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); }else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
   }

   startphi *= aunit;
   deltaphi *= aunit;

   std::vector<zplaneType> zplaneList;

   for (xercesc::DOMNode* iter = polyconeElement->getFirstChild();
        iter != 0; iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE) { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="zplane") { zplaneList.push_back(ZplaneRead(child)); }
   }

   G4int numZPlanes = zplaneList.size();

   G4double* rmin_array = new G4double[numZPlanes];
   G4double* rmax_array = new G4double[numZPlanes];
   G4double* z_array    = new G4double[numZPlanes];

   for (G4int i=0; i<numZPlanes; i++)
   { 
      rmin_array[i] = zplaneList[i].rmin*lunit;
      rmax_array[i] = zplaneList[i].rmax*lunit;
      z_array[i]    = zplaneList[i].z*lunit;
   }

   new G4Polycone(name,startphi,deltaphi,numZPlanes,
                  z_array,rmin_array,rmax_array);
}

void G4DAEReadSolids::
PolyhedraRead(const xercesc::DOMElement* const polyhedraElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;
   G4int numsides = 0;

   const xercesc::DOMNamedNodeMap* const attributes
         = polyhedraElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); } else
      if (attName=="numsides") { numsides = eval.EvaluateInteger(attValue); }
   }

   startphi *= aunit;
   deltaphi *= aunit;

   std::vector<zplaneType> zplaneList;

   for (xercesc::DOMNode* iter = polyhedraElement->getFirstChild();
        iter != 0; iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="zplane") { zplaneList.push_back(ZplaneRead(child)); }
   }

   G4int numZPlanes = zplaneList.size();

   G4double* rmin_array = new G4double[numZPlanes];
   G4double* rmax_array = new G4double[numZPlanes];
   G4double* z_array = new G4double[numZPlanes];

   for (G4int i=0; i<numZPlanes; i++)
   { 
      rmin_array[i] = zplaneList[i].rmin*lunit;
      rmax_array[i] = zplaneList[i].rmax*lunit;
      z_array[i] = zplaneList[i].z*lunit;
   }

   new G4Polyhedra(name,startphi,deltaphi,numsides,numZPlanes,
                   z_array,rmin_array,rmax_array);

}

G4QuadrangularFacet* G4DAEReadSolids::
QuadrangularRead(const xercesc::DOMElement* const quadrangularElement)
{
   G4ThreeVector vertex1;
   G4ThreeVector vertex2;
   G4ThreeVector vertex3;
   G4ThreeVector vertex4;
   G4FacetVertexType type = ABSOLUTE;

   const xercesc::DOMNamedNodeMap* const attributes
         = quadrangularElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="vertex1")
        { vertex1 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex2")
        { vertex2 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex3")
        { vertex3 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex4")
        { vertex4 = GetPosition(GenerateName(attValue)); } else
      if (attName=="type")
        { if (attValue=="RELATIVE") { type = RELATIVE; } }
   }

   return new G4QuadrangularFacet(vertex1,vertex2,vertex3,vertex4,type);
}

void G4DAEReadSolids::
ReflectedSolidRead(const xercesc::DOMElement* const reflectedSolidElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4String solid;
   G4ThreeVector scale(1.0,1.0,1.0);
   G4ThreeVector rotation;
   G4ThreeVector position;

   const xercesc::DOMNamedNodeMap* const attributes
         = reflectedSolidElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="solid") { solid = GenerateName(attValue); } else
      if (attName=="sx") { scale.setX(eval.Evaluate(attValue)); } else
      if (attName=="sy") { scale.setY(eval.Evaluate(attValue)); } else
      if (attName=="sz") { scale.setZ(eval.Evaluate(attValue)); } else
      if (attName=="rx") { rotation.setX(eval.Evaluate(attValue)); } else
      if (attName=="ry") { rotation.setY(eval.Evaluate(attValue)); } else
      if (attName=="rz") { rotation.setZ(eval.Evaluate(attValue)); } else
      if (attName=="dx") { position.setX(eval.Evaluate(attValue)); } else
      if (attName=="dy") { position.setY(eval.Evaluate(attValue)); } else
      if (attName=="dz") { position.setZ(eval.Evaluate(attValue)); }
   }

   rotation *= aunit;
   position *= lunit;

   G4Transform3D transform(GetRotationMatrix(rotation),position);
   transform = transform*G4Scale3D(scale.x(),scale.y(),scale.z());
          
   new G4ReflectedSolid(name,GetSolid(solid),transform);
}

G4ExtrudedSolid::ZSection G4DAEReadSolids::
SectionRead(const xercesc::DOMElement* const sectionElement,G4double lunit) 
{
   G4double zPosition = 0.0;
   G4TwoVector Offset;
   G4double scalingFactor = 1.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = sectionElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="zPosition")
        { zPosition = eval.Evaluate(attValue)*lunit; } else
      if (attName=="xOffset")
        { Offset.setX(eval.Evaluate(attValue)*lunit); } else
      if (attName=="yOffset")
        { Offset.setY(eval.Evaluate(attValue)*lunit); } else
      if (attName=="scalingFactor")
        { scalingFactor = eval.Evaluate(attValue); }
   }

   return G4ExtrudedSolid::ZSection(zPosition,Offset,scalingFactor);
}

void G4DAEReadSolids::
SphereRead(const xercesc::DOMElement* const sphereElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rmin = 0.0;
   G4double rmax = 0.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;
   G4double starttheta = 0.0;
   G4double deltatheta = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = sphereElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rmin") { rmin = eval.Evaluate(attValue); } else
      if (attName=="rmax") { rmax = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); } else
      if (attName=="starttheta") { starttheta = eval.Evaluate(attValue); } else
      if (attName=="deltatheta") { deltatheta = eval.Evaluate(attValue); }
   }

   rmin *= lunit;
   rmax *= lunit;
   startphi *= aunit;
   deltaphi *= aunit;
   starttheta *= aunit;
   deltatheta *= aunit;

   new G4Sphere(name,rmin,rmax,startphi,deltaphi,starttheta,deltatheta);
}

void G4DAEReadSolids::
TessellatedRead(const xercesc::DOMElement* const tessellatedElement)
{
   G4String name;

   const xercesc::DOMNamedNodeMap* const attributes
         = tessellatedElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); }
   }
   
   G4TessellatedSolid *tessellated = new G4TessellatedSolid(name);

   for (xercesc::DOMNode* iter = tessellatedElement->getFirstChild();
        iter != 0; iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE) { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="triangular")
        { tessellated->AddFacet(TriangularRead(child)); } else
      if (tag=="quadrangular")
        { tessellated->AddFacet(QuadrangularRead(child)); }
   }

   tessellated->SetSolidClosed(true);
}

void G4DAEReadSolids::TetRead(const xercesc::DOMElement* const tetElement)
{
   G4String name;
   G4ThreeVector vertex1;
   G4ThreeVector vertex2;
   G4ThreeVector vertex3;
   G4ThreeVector vertex4;
   
   const xercesc::DOMNamedNodeMap* const attributes
         = tetElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount;attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name")
        { name = GenerateName(attValue); } else
      if (attName=="vertex1")
        { vertex1 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex2")
        { vertex2 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex3")
        { vertex3 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex4")
        { vertex4 = GetPosition(GenerateName(attValue)); }
   }

   new G4Tet(name,vertex1,vertex2,vertex3,vertex4);
}

void G4DAEReadSolids::TorusRead(const xercesc::DOMElement* const torusElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rmin = 0.0;
   G4double rmax = 0.0;
   G4double rtor = 0.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = torusElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rmin") { rmin = eval.Evaluate(attValue); } else
      if (attName=="rmax") { rmax = eval.Evaluate(attValue); } else
      if (attName=="rtor") { rtor = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
   }

   rmin *= lunit;
   rmax *= lunit;
   rtor *= lunit;
   startphi *= aunit;
   deltaphi *= aunit;

   new G4Torus(name,rmin,rmax,rtor,startphi,deltaphi);
}

void G4DAEReadSolids::TrapRead(const xercesc::DOMElement* const trapElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double z = 0.0;
   G4double theta = 0.0;
   G4double phi = 0.0;
   G4double y1 = 0.0;
   G4double x1 = 0.0;
   G4double x2 = 0.0;
   G4double alpha1 = 0.0;
   G4double y2 = 0.0;
   G4double x3 = 0.0;
   G4double x4 = 0.0;
   G4double alpha2 = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = trapElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="theta") { theta = eval.Evaluate(attValue); } else
      if (attName=="phi") { phi = eval.Evaluate(attValue); } else
      if (attName=="y1") { y1 = eval.Evaluate(attValue); } else
      if (attName=="x1") { x1 = eval.Evaluate(attValue); } else
      if (attName=="x2") { x2 = eval.Evaluate(attValue); } else
      if (attName=="alpha1") { alpha1 = eval.Evaluate(attValue); } else
      if (attName=="y2") { y2 = eval.Evaluate(attValue); } else
      if (attName=="x3") { x3 = eval.Evaluate(attValue); } else
      if (attName=="x4") { x4 = eval.Evaluate(attValue); } else
      if (attName=="alpha2") { alpha2 = eval.Evaluate(attValue); }
   }

   z *= 0.5*lunit;
   theta *= aunit;
   phi *= aunit;
   y1 *= 0.5*lunit;
   x1 *= 0.5*lunit;
   x2 *= 0.5*lunit;
   alpha1 *= aunit;
   y2 *= 0.5*lunit;
   x3 *= 0.5*lunit;
   x4 *= 0.5*lunit;
   alpha2 *= aunit;

   new G4Trap(name,z,theta,phi,y1,x1,x2,alpha1,y2,x3,x4,alpha2);
}

void G4DAEReadSolids::TrdRead(const xercesc::DOMElement* const trdElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double x1 = 0.0;
   G4double x2 = 0.0;
   G4double y1 = 0.0;
   G4double y2 = 0.0;
   G4double z = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes = trdElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="x1") { x1 = eval.Evaluate(attValue); } else
      if (attName=="x2") { x2 = eval.Evaluate(attValue); } else
      if (attName=="y1") { y1 = eval.Evaluate(attValue); } else
      if (attName=="y2") { y2 = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); }
   }

   x1 *= 0.5*lunit;
   x2 *= 0.5*lunit;
   y1 *= 0.5*lunit;
   y2 *= 0.5*lunit;
   z *= 0.5*lunit;

   new G4Trd(name,x1,x2,y1,y2,z);
}

G4TriangularFacet* G4DAEReadSolids::
TriangularRead(const xercesc::DOMElement* const triangularElement)
{
   G4ThreeVector vertex1;
   G4ThreeVector vertex2;
   G4ThreeVector vertex3;
   G4FacetVertexType type = ABSOLUTE;

   const xercesc::DOMNamedNodeMap* const attributes
         = triangularElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="vertex1")
        { vertex1 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex2")
        { vertex2 = GetPosition(GenerateName(attValue)); } else
      if (attName=="vertex3")
        { vertex3 = GetPosition(GenerateName(attValue)); } else
      if (attName=="type")
        { if (attValue=="RELATIVE") { type = RELATIVE; } }
   }

   return new G4TriangularFacet(vertex1,vertex2,vertex3,type);
}

void G4DAEReadSolids::TubeRead(const xercesc::DOMElement* const tubeElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double rmin = 0.0;
   G4double rmax = 0.0;
   G4double z = 0.0;
   G4double startphi = 0.0;
   G4double deltaphi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = tubeElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="rmin") { rmin = eval.Evaluate(attValue); } else
      if (attName=="rmax") { rmax = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="startphi") { startphi = eval.Evaluate(attValue); } else
      if (attName=="deltaphi") { deltaphi = eval.Evaluate(attValue); }
   }

   rmin *= lunit;
   rmax *= lunit;
   z *= 0.5*lunit;
   startphi *= aunit;
   deltaphi *= aunit;

   new G4Tubs(name,rmin,rmax,z,startphi,deltaphi);
}

void G4DAEReadSolids::
TwistedboxRead(const xercesc::DOMElement* const twistedboxElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double PhiTwist = 0.0;
   G4double x = 0.0;
   G4double y = 0.0;
   G4double z = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = twistedboxElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="PhiTwist") { PhiTwist = eval.Evaluate(attValue); } else
      if (attName=="x") { x = eval.Evaluate(attValue); } else
      if (attName=="y") { y = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); }
   }

   PhiTwist *= aunit;
   x *= 0.5*lunit;
   y *= 0.5*lunit;
   z *= 0.5*lunit;

   new G4TwistedBox(name,PhiTwist,x,y,z);
}

void G4DAEReadSolids::
TwistedtrapRead(const xercesc::DOMElement* const twistedtrapElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double PhiTwist = 0.0;
   G4double z = 0.0;
   G4double Theta = 0.0;
   G4double Phi = 0.0;
   G4double y1 = 0.0;
   G4double x1 = 0.0;
   G4double x2 = 0.0;
   G4double y2 = 0.0;
   G4double x3 = 0.0;
   G4double x4 = 0.0;
   G4double Alph = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = twistedtrapElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="PhiTwist") { PhiTwist = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="Theta") { Theta = eval.Evaluate(attValue); } else
      if (attName=="Phi") { Phi = eval.Evaluate(attValue); } else
      if (attName=="y1") { y1 = eval.Evaluate(attValue); } else
      if (attName=="x1") { x1 = eval.Evaluate(attValue); } else
      if (attName=="x2") { x2 = eval.Evaluate(attValue); } else
      if (attName=="y2") { y2 = eval.Evaluate(attValue); } else
      if (attName=="x3") { x3 = eval.Evaluate(attValue); } else
      if (attName=="x4") { x4 = eval.Evaluate(attValue); } else
      if (attName=="Alph") { Alph = eval.Evaluate(attValue); }
   }


   PhiTwist *= aunit;
   z *= 0.5*lunit;
   Theta *= aunit;
   Phi *= aunit;
   Alph *= aunit;
   y1 *= 0.5*lunit;
   x1 *= 0.5*lunit;
   x2 *= 0.5*lunit;
   y2 *= 0.5*lunit;
   x3 *= 0.5*lunit;
   x4 *= 0.5*lunit;

   new G4TwistedTrap(name,PhiTwist,z,Theta,Phi,y1,x1,x2,y2,x3,x4,Alph);
}

void G4DAEReadSolids::
TwistedtrdRead(const xercesc::DOMElement* const twistedtrdElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double x1 = 0.0;
   G4double x2 = 0.0;
   G4double y1 = 0.0;
   G4double y2 = 0.0;
   G4double z = 0.0;
   G4double PhiTwist = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = twistedtrdElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="x1") { x1 = eval.Evaluate(attValue); } else
      if (attName=="x2") { x2 = eval.Evaluate(attValue); } else
      if (attName=="y1") { y1 = eval.Evaluate(attValue); } else
      if (attName=="y2") { y2 = eval.Evaluate(attValue); } else
      if (attName=="z") { z = eval.Evaluate(attValue); } else
      if (attName=="PhiTwist") { PhiTwist = eval.Evaluate(attValue); }
   }

   x1 *= 0.5*lunit;
   x2 *= 0.5*lunit;
   y1 *= 0.5*lunit;
   y2 *= 0.5*lunit;
   z *= 0.5*lunit;
   PhiTwist *= aunit;

   new G4TwistedTrd(name,x1,x2,y1,y2,z,PhiTwist);
}

void G4DAEReadSolids::
TwistedtubsRead(const xercesc::DOMElement* const twistedtubsElement)
{
   G4String name;
   G4double lunit = 1.0;
   G4double aunit = 1.0;
   G4double twistedangle = 0.0;
   G4double endinnerrad = 0.0;
   G4double endouterrad = 0.0;
   G4double zlen = 0.0;
   G4double phi = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = twistedtubsElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="lunit") { lunit = eval.Evaluate(attValue); } else
      if (attName=="aunit") { aunit = eval.Evaluate(attValue); } else
      if (attName=="twistedangle") { twistedangle=eval.Evaluate(attValue); } else
      if (attName=="endinnerrad")  { endinnerrad=eval.Evaluate(attValue);  } else
      if (attName=="endouterrad")  { endouterrad=eval.Evaluate(attValue);  } else
      if (attName=="zlen") { zlen = eval.Evaluate(attValue); } else
      if (attName=="phi") { phi = eval.Evaluate(attValue); }
   }

   twistedangle *= aunit;
   endinnerrad *= lunit;
   endouterrad *= lunit;
   zlen *= 0.5*lunit;
   phi *= aunit;

   new G4TwistedTubs(name,twistedangle,endinnerrad,endouterrad,zlen,phi);
}

G4TwoVector G4DAEReadSolids::
TwoDimVertexRead(const xercesc::DOMElement* const element, G4double lunit)
{
   G4TwoVector vec;
   
   const xercesc::DOMNamedNodeMap* const attributes = element->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="x") { vec.setX(eval.Evaluate(attValue)*lunit); } else
      if (attName=="y") { vec.setY(eval.Evaluate(attValue)*lunit); }
   }

   return vec;
}

G4DAEReadSolids::zplaneType G4DAEReadSolids::
ZplaneRead(const xercesc::DOMElement* const zplaneElement)
{
   zplaneType zplane;

   const xercesc::DOMNamedNodeMap* const attributes
         = zplaneElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* node = attributes->item(attribute_index);

      if (node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE) { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="rmin") { zplane.rmin = eval.Evaluate(attValue); } else
      if (attName=="rmax") { zplane.rmax = eval.Evaluate(attValue); } else
      if (attName=="z") { zplane.z = eval.Evaluate(attValue); }
   }

   return zplane;
}

void G4DAEReadSolids::
OpticalsurfaceRead(const xercesc::DOMElement* const opticalsurfaceElement)
{
   G4String name;
   G4String smodel;
   G4String sfinish;
   G4String stype;
   G4double value = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = opticalsurfaceElement->getAttributes();
   XMLSize_t attributeCount = attributes->getLength();

   for (XMLSize_t attribute_index=0;
        attribute_index<attributeCount; attribute_index++)
   {
      xercesc::DOMNode* attribute_node = attributes->item(attribute_index);

      if (attribute_node->getNodeType() != xercesc::DOMNode::ATTRIBUTE_NODE)
        { continue; }

      const xercesc::DOMAttr* const attribute
            = dynamic_cast<xercesc::DOMAttr*>(attribute_node);   
      const G4String attName = Transcode(attribute->getName());
      const G4String attValue = Transcode(attribute->getValue());

      if (attName=="name") { name = GenerateName(attValue); } else
      if (attName=="model") { smodel = attValue; } else
      if (attName=="finish") { sfinish = attValue; } else
      if (attName=="type") { stype = attValue; } else
      if (attName=="value") { value = eval.Evaluate(attValue); }
   }

   G4OpticalSurfaceModel model; 
   G4OpticalSurfaceFinish finish;
   G4SurfaceType type;   
   
   if (smodel="unified") { model = unified; } else { model = glisur; }

   if (sfinish=="polishedfrontpainted") { finish = polishedfrontpainted; } else
   if (sfinish=="polishedbackpainted") { finish = polishedbackpainted; } else
   if (sfinish=="groundfrontpainted") { finish = groundfrontpainted; } else
   if (sfinish=="groundbackpainted") { finish = groundbackpainted; } else
   if (sfinish=="ground") { finish = ground; } else { finish = polished; }

   if (stype=="dielectric_metal") { type = dielectric_metal; } else
   if (stype=="x_ray") { type = x_ray; } else
   if (stype=="firsov") { type = firsov; } else { type = dielectric_dielectric; }

   new G4OpticalSurface(name,model,finish,type,value);
}

void G4DAEReadSolids::SolidsRead(const xercesc::DOMElement* const solidsElement)
{
   G4cout << "G4DAE: Reading solids..." << G4endl;

   for (xercesc::DOMNode* iter = solidsElement->getFirstChild();
        iter != 0; iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="box") { BoxRead(child); } else
      if (tag=="cone") { ConeRead(child); } else
      if (tag=="elcone") { ElconeRead(child); } else
      if (tag=="ellipsoid") { EllipsoidRead(child); }else
      if (tag=="eltube") { EltubeRead(child); } else
      if (tag=="xtru") { XtruRead(child); } else
      if (tag=="hype") { HypeRead(child); } else
      if (tag=="intersection") { BooleanRead(child,INTERSECTION); } else
      if (tag=="orb") { OrbRead(child); } else
      if (tag=="para") { ParaRead(child); } else
      if (tag=="paraboloid") { ParaboloidRead(child); } else
      if (tag=="polycone") { PolyconeRead(child); } else
      if (tag=="polyhedra") { PolyhedraRead(child); } else
      if (tag=="reflectedSolid") { ReflectedSolidRead(child); } else
      if (tag=="sphere") { SphereRead(child); } else
      if (tag=="subtraction") { BooleanRead(child,SUBTRACTION); } else
      if (tag=="tessellated") { TessellatedRead(child); } else
      if (tag=="tet") { TetRead(child); } else
      if (tag=="torus") { TorusRead(child); } else
      if (tag=="trap") { TrapRead(child); } else
      if (tag=="trd") { TrdRead(child); } else
      if (tag=="tube") { TubeRead(child); } else
      if (tag=="twistedbox") { TwistedboxRead(child); } else
      if (tag=="twistedtrap") { TwistedtrapRead(child); } else
      if (tag=="twistedtrd") { TwistedtrdRead(child); } else
      if (tag=="twistedtubs") { TwistedtubsRead(child); } else
      if (tag=="union") { BooleanRead(child,UNION); } else
      if (tag=="opticalsurface") { OpticalsurfaceRead(child); } else
      if (tag=="loop") { LoopRead(child,&G4DAERead::SolidsRead); }
      else
      {
        G4String error_msg = "Unknown tag in solids: " + tag;
        G4Exception("G4DAEReadSolids::SolidsRead()", "ReadError",
                    FatalException, error_msg);
      }
   }
}

G4VSolid* G4DAEReadSolids::GetSolid(const G4String& ref) const
{
   G4VSolid* solidPtr = G4SolidStore::GetInstance()->GetSolid(ref,false);

   if (!solidPtr)
   {
     G4String error_msg = "Referenced solid '" + ref + "' was not found!";
     G4Exception("G4DAEReadSolids::GetSolid()", "ReadError",
                 FatalException, error_msg);
   }

   return solidPtr;
}

G4SurfaceProperty* G4DAEReadSolids::
GetSurfaceProperty(const G4String& ref) const
{
   const G4SurfacePropertyTable* surfaceList
         = G4SurfaceProperty::GetSurfacePropertyTable();
   const size_t surfaceCount = surfaceList->size();

   for (size_t i=0; i<surfaceCount; i++)
   {
      if ((*surfaceList)[i]->GetName() == ref)  { return (*surfaceList)[i]; }
   }

   G4String error_msg = "Referenced optical surface '" + ref + "' was not found!";
   G4Exception("G4DAEReadSolids::GetSurfaceProperty()", "ReadError",
               FatalException, error_msg);

   return 0;
}
