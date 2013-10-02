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

void visit( G4VPhysicalVolume* pv, G4LogicalVolume* lv, G4int nd)
{
  G4cout << pv->GetName() << " " << lv->GetName() << " " <<  nd << G4endl ;
  G4VSolid* vso = lv->GetSolid();
  G4Polyhedron* polyhedron = vso->GetPolyhedron();
  G4int nvert =  polyhedron->GetNoVertices();
  G4cout << vso->GetName() << " " << nvert << G4endl ;
  G4int i, j;
  for (i = 1, j = nvert; j; j--, i++) {
      G4Point3D point = polyhedron->GetVertex(i);
      G4cout << point << G4endl ;
  }
}

void recurse(G4VPhysicalVolume* pv)
{
   G4LogicalVolume* lv = pv->GetLogicalVolume();
   G4int nd = lv->GetNoDaughters();
   visit(pv,lv,nd);
   for (G4int i = 0; i < nd; i++) {
        G4VPhysicalVolume* dpv = lv->GetDaughter(i);
        recurse(dpv);
   }
}

int main(int argc, char** argv)
{
   G4GDMLParser parser;
   parser.Read("/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml", false);
   recurse(parser.GetWorldVolume()); 
}


