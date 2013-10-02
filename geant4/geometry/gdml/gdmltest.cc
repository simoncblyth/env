/*

gdmltest
--------

Build with env bash function gdml-;gdml-test

Succeeding to access the vertices unlike g4gdml.py the Geant4Py boost_python sibling of this

NEXT: 

  #. get this working together with collada dom, OR raw xerces-c xml creation 
  #. also could try to add polyhedon access to g4py in order to use PyCollada 

*/
#include "G4GDMLParser.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4VSolid.hh"
#include "G4Polyhedron.hh"
#include "G4Point3D.hh"
#include "G4Transform3D.hh"

#include "G4VisAttributes.hh"

#include <fstream>


class Traverse
{
  public:  

     Traverse();
     ~Traverse();
     void Read(const G4String& filename, G4bool validate);
     void Write(const G4String& filename);

     void Recurse(G4VPhysicalVolume* pv,const G4Transform3D& theAT );
     void Visit(G4VPhysicalVolume* pv, G4LogicalVolume* lv, G4int nd);
     void VisitPolyhedron( const G4Polyhedron& polyhedron );
     void PreAddSolid(const G4Transform3D& objectTransformation, const G4VisAttributes& visAttribs);
     void PostAddSolid();

  private:

    std::ofstream   fDest ;
    G4GDMLParser fParser ; 
    G4int fDepth  ;

    //  geant4.10.00.b01/source/visualization/management/include/G4VSceneHandler.hh
 
    G4bool             fProcessingSolid; // True if within Pre/PostAddSolid.
    G4VModel*          fpModel;          // Current model.
    G4Transform3D fObjectTransformation; // Current accumulated

    const G4VisAttributes* fpVisAttribs;  
    const G4Transform3D fIdentityTransformation;

};


Traverse::Traverse() : 
    fDest                  (),
    fProcessingSolid       (false),
    fpVisAttribs           (0),
    fpModel                (0)
{
}

void Traverse::Read(const G4String& filename, G4bool validate)
{
   G4cout << "Traverse::Read " << filename << " validate: " << validate << G4endl ;
   fParser.Read(filename,validate);
}
void Traverse::Write(const G4String& filename)
{
   G4cout << "Traverse::Write " << filename << G4endl ;
   fDest.open(filename);
   Recurse(fParser.GetWorldVolume()); 
   fDest.close();
}


// geant4.10.00.b01/source/visualization/management/src/G4VSceneHandler.cc
void Traverse::PreAddSolid (const G4Transform3D& objectTransformation, const G4VisAttributes& visAttribs) 
{
   fObjectTransformation = objectTransformation;
   fpVisAttribs = &visAttribs;
   fProcessingSolid = true;
}

void Traverse::PostAddSolid () {
  fpVisAttribs = 0;
  fProcessingSolid = false;
}


Traverse::~Traverse()
{
}

void Traverse::Visit( G4VPhysicalVolume* pVPV, G4LogicalVolume* lv, G4int nd)
{

  //  $DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc
  const G4RotationMatrix objectRotation = pVPV -> GetObjectRotationValue ();
  const G4ThreeVector&  translation     = pVPV -> GetTranslation ();
  G4Transform3D theLT (G4Transform3D (objectRotation, translation));

  G4Transform3D theNewAT (theAT);

  if (fCurrentDepth != 0) theNewAT = theAT * theLT;



  G4cout << pVPV->GetName() << " " << lv->GetName() << " " <<  nd << G4endl ;
  G4VSolid* vso = lv->GetSolid();
  G4Polyhedron* pPolyhedron = vso->GetPolyhedron();
  VisitPolyhedron(*pPolyhedron);
}


void Traverse::VisitPolyhedron( const G4Polyhedron& polyhedron )
{
    // geant4.10.00.b01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc

    // Current Model
    //const G4VModel* pv_model  = GetModel();
    //G4String pv_name = "No model";
    //if (pv_model) pv_name = pv_model->GetCurrentTag() ;

    // VRML codes are generated below

    //fDest << "#---------- SOLID: " << pv_name << "\n";

    fDest << "\t"; fDest << "Shape {" << "\n";

    //SendMaterialNode();

    fDest << "\t\t" << "geometry IndexedFaceSet {" << "\n";

    fDest << "\t\t\t"   << "coord Coordinate {" << "\n";
    fDest << "\t\t\t\t" <<      "point [" << "\n";
    G4int i, j;
    for (i = 1, j = polyhedron.GetNoVertices(); j; j--, i++) {
        G4Point3D point = polyhedron.GetVertex(i);

        //point.transform( fObjectTransformation );

        fDest << "\t\t\t\t\t";
        fDest <<                   point.x() << " ";
        fDest <<                   point.y() << " ";
        fDest <<                   point.z() << "," << "\n";
    }   
    fDest << "\t\t\t\t" <<      "]" << "\n"; // point
    fDest << "\t\t\t"   << "}"      << "\n"; // coord

    fDest << "\t\t\t"   << "coordIndex [" << "\n";

    // facet loop   
    G4int f;
    for (f = polyhedron.GetNoFacets(); f; f--) {

        // edge loop  
        G4bool notLastEdge;
        G4int index = -1, edgeFlag = 1;
        fDest << "\t\t\t\t";
        do {
            notLastEdge = polyhedron.GetNextVertexIndex(index, edgeFlag);
            fDest << index - 1 << ", ";
        } while (notLastEdge);
        fDest << "-1," << "\n";
    }   
}


void Traverse::Recurse(G4VPhysicalVolume* pv, const G4Transform3D& theAT )
{
   G4LogicalVolume* lv = pv->GetLogicalVolume();
   G4int nd = lv->GetNoDaughters();
   Visit(pv,lv,nd);
   for (G4int i = 0; i < nd; i++) {
        G4VPhysicalVolume* dpv = lv->GetDaughter(i);
        Recurse(dpv);
   }
}



int main(int argc, char** argv)
{
   Traverse t ;
   t.Read("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml", false);
   t.Write("test.wrl");
}


