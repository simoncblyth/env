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
// $Id: G4DAEReadDefine.cc,v 1.20 2008/07/16 15:46:34 gcosmo Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
// class G4DAEReadDefine Implementation
//
// Original author: Zoltan Torzsok, November 2007
//
// --------------------------------------------------------------------

#include "G4DAEReadDefine.hh"

G4DAEMatrix::G4DAEMatrix()
{
   rows = 0;
   cols = 0;
   m = 0;
}

G4DAEMatrix::G4DAEMatrix(size_t rows0,size_t cols0)
{   
   rows = rows0;
   cols = cols0;
   m = new G4double[rows*cols];
}

G4DAEMatrix::~G4DAEMatrix()
{
   if (m) { delete [] m; }
}

void G4DAEMatrix::Set(size_t r,size_t c,G4double a)
{   
   if (r>=rows || c>=cols)
   {
     G4Exception("G4DAEMatrix::set()", "InvalidSetup",
                 FatalException, "Index out of range!");
   }
   m[cols*r+c] = a;
}

G4double G4DAEMatrix::Get(size_t r,size_t c) const
{   
   if (r>=rows || c>=cols)
   {
     G4Exception("G4DAEMatrix::get()", "InvalidSetup",
                 FatalException, "Index out of range!");
   }
   return m[cols*r+c];
}

size_t G4DAEMatrix::GetRows() const
{
   return rows;
}

size_t G4DAEMatrix::GetCols() const
{
   return cols;
}

G4RotationMatrix
G4DAEReadDefine::GetRotationMatrix(const G4ThreeVector& angles)
{
   G4RotationMatrix rot;

   rot.rotateX(angles.x());
   rot.rotateY(angles.y());
   rot.rotateZ(angles.z());

   return rot;
}

void
G4DAEReadDefine::ConstantRead(const xercesc::DOMElement* const constantElement)
{
   G4String name  = "";
   G4double value = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = constantElement->getAttributes();
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

      if (attName=="name")  { name = attValue; }  else
      if (attName=="value") { value = eval.Evaluate(attValue); }
   }

   eval.DefineConstant(name,value);
}

void
G4DAEReadDefine::MatrixRead(const xercesc::DOMElement* const matrixElement) 
{
   G4String name = "";
   G4int coldim  = 0;
   G4String values = "";

   const xercesc::DOMNamedNodeMap* const attributes
         = matrixElement->getAttributes();
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

      if (attName=="name")   { name  = GenerateName(attValue); } else
      if (attName=="coldim") { coldim = eval.EvaluateInteger(attValue); } else
      if (attName=="values") { values = attValue; }
   }

   std::stringstream MatrixValueStream(values);
   std::vector<G4double> valueList;

   while (!MatrixValueStream.eof())
   {
      G4String MatrixValue;
      MatrixValueStream >> MatrixValue;
      valueList.push_back(eval.Evaluate(MatrixValue));
   }

   eval.DefineMatrix(name,coldim,valueList);

   G4DAEMatrix matrix(valueList.size()/coldim,coldim);

   for (size_t i=0;i<valueList.size();i++)
   {
      matrix.Set(i/coldim,i%coldim,valueList[i]);
   }

   matrixMap[name] = matrix;
}

void
G4DAEReadDefine::PositionRead(const xercesc::DOMElement* const positionElement)
{
   G4String name = "";
   G4double unit = 1.0;
   G4ThreeVector position(0.,0.,0.);

   const xercesc::DOMNamedNodeMap* const attributes
         = positionElement->getAttributes();
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

      if (attName=="name") { name = GenerateName(attValue); }  else
      if (attName=="unit") { unit = eval.Evaluate(attValue); } else
      if (attName=="x") { position.setX(eval.Evaluate(attValue)); } else
      if (attName=="y") { position.setY(eval.Evaluate(attValue)); } else
      if (attName=="z") { position.setZ(eval.Evaluate(attValue)); }
   }

   positionMap[name] = position*unit;
}

void
G4DAEReadDefine::RotationRead(const xercesc::DOMElement* const rotationElement)
{
   G4String name = "";
   G4double unit = 1.0;
   G4ThreeVector rotation(0.,0.,0.);

   const xercesc::DOMNamedNodeMap* const attributes
         = rotationElement->getAttributes();
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

      if (attName=="name") { name = GenerateName(attValue); }  else
      if (attName=="unit") { unit = eval.Evaluate(attValue); } else
      if (attName=="x") { rotation.setX(eval.Evaluate(attValue)); } else
      if (attName=="y") { rotation.setY(eval.Evaluate(attValue)); } else
      if (attName=="z") { rotation.setZ(eval.Evaluate(attValue)); }
   }

   rotationMap[name] = rotation*unit;
}

void G4DAEReadDefine::ScaleRead(const xercesc::DOMElement* const scaleElement)
{
   G4String name = "";
   G4ThreeVector scale(1.0,1.0,1.0);

   const xercesc::DOMNamedNodeMap* const attributes
         = scaleElement->getAttributes();
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

      if (attName=="name") { name = GenerateName(attValue); }    else
      if (attName=="x") { scale.setX(eval.Evaluate(attValue)); } else
      if (attName=="y") { scale.setY(eval.Evaluate(attValue)); } else
      if (attName=="z") { scale.setZ(eval.Evaluate(attValue)); }
   }

   scaleMap[name] = scale;
}

void
G4DAEReadDefine::VariableRead(const xercesc::DOMElement* const variableElement)
{
   G4String name  = "";
   G4double value = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = variableElement->getAttributes();
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

      if (attName=="name")  { name = attValue; } else
      if (attName=="value") { value = eval.Evaluate(attValue); }
   }

   eval.DefineVariable(name,value);
}

void G4DAEReadDefine::QuantityRead(const xercesc::DOMElement* const element)
{
   G4String name = "";
   G4double unit = 1.0;
   G4double value = 0.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = element->getAttributes();
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

      if (attName=="name") { name = attValue; } else
      if (attName=="value") { value = eval.Evaluate(attValue); } else
      if (attName=="unit") { unit = eval.Evaluate(attValue); }
   }

   quantityMap[name] = value*unit;
}

void
G4DAEReadDefine::DefineRead(const xercesc::DOMElement* const defineElement)
{
   G4cout << "G4DAE: Reading definitions..." << G4endl;

   for (xercesc::DOMNode* iter = defineElement->getFirstChild();
        iter != 0;iter = iter->getNextSibling())
   {
      if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE) { continue; }

      const xercesc::DOMElement* const child
            = dynamic_cast<xercesc::DOMElement*>(iter);
      const G4String tag = Transcode(child->getTagName());

      if (tag=="constant") { ConstantRead(child); } else
      if (tag=="matrix")   { MatrixRead(child); }   else
      if (tag=="position") { PositionRead(child); } else
      if (tag=="rotation") { RotationRead(child); } else
      if (tag=="scale")    { ScaleRead(child); }    else
      if (tag=="variable") { VariableRead(child); } else
      if (tag=="quantity") { QuantityRead(child); }
      else
      {
        G4String error_msg = "Unknown tag in define: "+tag;
        G4Exception("G4DAEReadDefine::defineRead()", "ReadError",
                    FatalException, error_msg);
      }
   }
}

void
G4DAEReadDefine::VectorRead(const xercesc::DOMElement* const vectorElement,
                             G4ThreeVector& vec)
{
   G4double unit = 1.0;

   const xercesc::DOMNamedNodeMap* const attributes
         = vectorElement->getAttributes();
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

      if (attName=="unit") { unit = eval.Evaluate(attValue); } else
      if (attName=="x") { vec.setX(eval.Evaluate(attValue)); } else
      if (attName=="y") { vec.setY(eval.Evaluate(attValue)); } else
      if (attName=="z") { vec.setZ(eval.Evaluate(attValue)); }
   }

   vec *= unit;
}

G4String G4DAEReadDefine::RefRead(const xercesc::DOMElement* const element)
{
   G4String ref;

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

      if (attName=="ref") { ref = attValue; }
   }

   return ref;
}

G4double G4DAEReadDefine::GetConstant(const G4String& ref)
{
   return eval.GetConstant(ref);
}

G4double G4DAEReadDefine::GetVariable(const G4String& ref)
{
   return eval.GetVariable(ref);
}

G4double G4DAEReadDefine::GetQuantity(const G4String& ref)
{
   if (quantityMap.find(ref) == quantityMap.end())
   {
     G4String error_msg = "Quantity '"+ref+"' was not found!";
     G4Exception("G4DAEReadDefine::getQuantity()", "ReadError",
                 FatalException, error_msg);
   }
   return quantityMap[ref];
}

G4ThreeVector G4DAEReadDefine::GetPosition(const G4String& ref)
{
   if (positionMap.find(ref) == positionMap.end())
   {
     G4String error_msg = "Position '"+ref+"' was not found!";
     G4Exception("G4DAEReadDefine::getPosition()", "ReadError",
                 FatalException, error_msg);
   }
   return positionMap[ref];
}

G4ThreeVector G4DAEReadDefine::GetRotation(const G4String& ref)
{
   if (rotationMap.find(ref) == rotationMap.end())
   {
     G4String error_msg = "Rotation '"+ref+"' was not found!";
     G4Exception("G4DAEReadDefine::getRotation()", "ReadError",
                 FatalException, error_msg);
   }
   return rotationMap[ref];
}

G4ThreeVector G4DAEReadDefine::GetScale(const G4String& ref)
{
   if (scaleMap.find(ref) == scaleMap.end())
   {
     G4String error_msg = "Scale '"+ref+"' was not found!";
     G4Exception("G4DAEReadDefine::getScale()", "ReadError",
                 FatalException, error_msg);
   }
   return scaleMap[ref];
}

G4DAEMatrix G4DAEReadDefine::GetMatrix(const G4String& ref)
{
   if (matrixMap.find(ref) == matrixMap.end())
   {
     G4String error_msg = "Matrix '"+ref+"' was not found!";
     G4Exception("G4DAEReadDefine::getMatrix()", "ReadError",
                 FatalException, error_msg);
   }
   return matrixMap[ref];
}
