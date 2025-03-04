#include "G4DAEPolyhedron.hh"

G4DAEPolyhedron::G4DAEPolyhedron( const G4Polyhedron& polyhedron)
{
    fStart = "\n" ;
    fBefItem  = "\t\t\t\t" ;
    fAftItem  = "\n" ;
    fEnd   = "" ;

    Vertices(polyhedron);   
    Normals(polyhedron);   

}

void G4DAEPolyhedron::Vertices( const G4Polyhedron& polyhedron )
{
    G4int nvert = polyhedron.GetNoVertices();
    G4int ivert, jvert ;

    std::ostringstream ss ;
    ss << fStart ; 
    for (ivert = 1, jvert = nvert ; jvert; jvert--, ivert++) {
        G4Point3D vtx = polyhedron.GetVertex(ivert);
        ss << fBefItem ; 
        ss << vtx.x() << " " ; 
        ss << vtx.y() << " " ; 
        ss << vtx.z() << " " ; 
        ss << fAftItem ; 
    }   
    ss << fEnd ; 
    fVertices = ss.str() ;
} 

void G4DAEPolyhedron::Dump()
{
    G4cout << "G4DAEPolyhedron Vertices " << fVertices << G4endl ;
    G4cout << "G4DAEPolyhedron Normals " << fNormals << G4endl ;
    G4cout << "G4DAEPolyhedron Facets" << G4endl ;
    for(std::vector<std::string>::iterator it = fFacets.begin(); it != fFacets.end(); ++it) {
         G4cout << *it << G4endl ; 
    }
}

void G4DAEPolyhedron::Normals( const G4Polyhedron& polyhedron )
{
    G4int nface = polyhedron.GetNoFacets();
    G4int iface, jface ;

    G4Normal3D norm ;
    std::ostringstream ss ;
    ss << fStart ; 

    for (iface = 1, jface = nface; jface; jface--, iface++ ) 
    {
        norm = polyhedron.GetUnitNormal(iface);  
        ss << fBefItem  ; 
        ss << norm.x() << " " ; 
        ss << norm.y() << " " ; 
        ss << norm.z() << " " ; 
        ss << fAftItem  ; 

        Facet( polyhedron, iface ) ;
    }          
    ss << fEnd ; 
    fNormals = ss.str() ;
}


void G4DAEPolyhedron::Facet( const G4Polyhedron& polyhedron, G4int iface )
{

    G4int nedge;
    G4int ivertex[4];
    G4int iedgeflag[4];
    G4int ifacet[4];
    polyhedron.GetFacet(iface, nedge, ivertex, iedgeflag, ifacet);
    std::ostringstream ss ;
    G4int iedge = 0;
    for(iedge = 0; iedge < nedge; ++iedge) {
        ss << ivertex[iedge] << " " << iface << "  " ;  
    }
    ss << " " ;
    std::string facet = ss.str() ; 
    fFacets.push_back( facet );
}




