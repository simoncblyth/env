/*
*/

#include "G4Polyhedron.hh"
#include "G4Point3D.hh"
#include "G4Normal3D.hh"
#include <iostream>
#include <iomanip>

using namespace std ; 




void facets2( const G4Polyhedron& polyhedron )
{
    //  source/visualization/OpenGL/src/G4OpenGLSceneHandler.cc
    G4int iface, jface ;
    G4int nface = polyhedron.GetNoFacets();

    G4Point3D vtx ;
    HepGeom::Normal3D<double> norm ;

    //First, find vertices, edgeflags and normals and note "not last facet"...
    G4int nedge;
    G4int ivertex[4];
    G4int iedgeflag[4];
    G4int ifacet[4];
    
    for (iface = 1, jface = nface; jface; jface--, iface++ ) 
    {
         norm = polyhedron.GetUnitNormal(iface);  
         G4cout << "F" << iface << " " << norm << G4endl ;
         polyhedron.GetFacet(iface, nedge, ivertex, iedgeflag, ifacet);
         G4int edgeCount = 0;
         for(edgeCount = 0; edgeCount < nedge; ++edgeCount) {

             vtx = polyhedron.GetVertex(ivertex[edgeCount]);

             G4cout << " iedgeflag[" << edgeCount << "] " << iedgeflag[edgeCount]  ;
             G4cout << " ivertex[" << edgeCount << "] " << ivertex[edgeCount] << " " << vtx ;
             G4cout << " ifacet[" << edgeCount << "] " << ifacet[edgeCount]  ;
             G4cout << G4endl ;
         }   
    }
}


void facets( const G4Polyhedron& polyhedron )
{
    //  source/visualization/OpenGL/src/G4OpenGLSceneHandler.cc
    G4int iface, jface ;
    G4int nface = polyhedron.GetNoFacets();

    //First, find vertices, edgeflags and normals and note "not last facet"...
    G4int nedge;
    G4Point3D vertex[4];
    G4int edgeFlag[4];
    G4Normal3D normals[4];
    
    for (iface = 1, jface = nface; jface; jface--, iface++ ) 
    {
         G4cout << "F" << iface << G4endl ;
         polyhedron.GetFacet(iface, nedge, vertex, edgeFlag, normals);
         //Loop through the four (or three) edges of each G4Facet...
         G4int edgeCount = 0;
         for(edgeCount = 0; edgeCount < nedge; ++edgeCount) {
             G4cout << "E" << edgeCount << " " << edgeFlag[edgeCount]  ;
             G4cout << "N" << edgeCount << " " << normals[edgeCount]  ;
             G4cout << "V" << edgeCount << " " << vertex[edgeCount]  ;
             G4cout << G4endl ;
         }   
    }
}

void vertices( const G4Polyhedron& polyhedron )
{
    G4int nvert = polyhedron.GetNoVertices();
    G4int nface = polyhedron.GetNoFacets();
    G4Point3D vtx ;
    HepGeom::Normal3D<double> norm ;

    G4int ivert, jvert ;
    G4cout << " nvert: " << nvert << G4endl ;
    for (ivert = 1, jvert = nvert ; jvert; jvert--, ivert++) {
        vtx = polyhedron.GetVertex(ivert);
        G4cout << " V" << ivert << "/" << jvert << " " << vtx << G4endl ;     
    }   

    G4cout << " nface: " << nface << G4endl ;
    G4int iface, jface ;
    for (iface = 1, jface = nface; jface; jface--, iface++ ) {
       G4bool notLastEdge;
       norm = polyhedron.GetUnitNormal(iface);  
       cout << "F" << iface << "/" << jface << "  " ;
       G4int index = -1, edgeFlag = 1;
       do {
            notLastEdge = polyhedron.GetNextVertexIndex(index, edgeFlag);
            vtx = polyhedron.GetVertex(index);
            cout << setw(5) << "V" << index << " " ; // << vtx ;
       } while (notLastEdge);
       cout << "       " << "N" << iface << " " << norm ;
       cout << endl ;
   }

}

void glean(){
    G4cout << ""
"face(1)= 1/2 4/3 3/4 2/5   \n"
"face(2)= 5/6 8/3 4/1 1/5   \n"
"face(3)= 8/6 7/4 3/1 4/2   \n"
"face(4)= 7/6 6/5 2/1 3/3   \n"
"face(5)= 6/6 5/2 1/1 2/4   \n"
"face(6)= 5/5 6/4 7/3 8/2   \n"
"                                               \n" 
"  Z     Y                                      \n" 
"   |   /                                       \n"
"   |  /                                        \n"
"   | /                                         \n"
"   *----- X                                    \n" 
"                 V8  +------------+ V7          \n"
"                    /            /             \n"
"                   /  F6 (+Z)   /              \n"
"                  /            /               \n"
"              V5 +------------+ V6              \n"
"                                                  \n"
"   (-X) F2                *              F4 (+X)       \n"
"                  V4 +------------+ V3             \n"
"                    /            /                 \n"
"                   /  F1 (-Z)   /                  \n"
"                  /            /                   \n"
"             V1  +------------+  V2                \n"
"                                                  \n"
"                                                  \n"
"                                                  \n"
"                                                   \n" << G4endl ; 
}



int main()
{
    G4PolyhedronBox polyhedron(10.,10.,10.);
    G4cout << polyhedron << G4endl ; 

    vertices(polyhedron);
    //facets(polyhedron);
    facets2(polyhedron);
    glean();
}
