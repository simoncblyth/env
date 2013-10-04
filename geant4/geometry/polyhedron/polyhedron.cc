#include "G4Polyhedron.hh"
#include "G4DAEPolyhedron.hh"

int main()
{
    G4PolyhedronBox polyhedron(10.,10.,10.);
    G4cout << polyhedron << G4endl ; 
    G4DAEPolyhedron p(polyhedron);
    p.Dump();
}



