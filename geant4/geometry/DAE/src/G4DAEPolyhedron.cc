#include "G4DAEPolyhedron.hh"
#include "G4VisAttributes.hh"   // for informative streaming 
#include "G4DAEUtil.hh"
#include "G4VSolid.hh"

typedef std::pair<std::string, std::string> KV ;

G4DAEPolyhedron::G4DAEPolyhedron( const G4VSolid* const solid, const G4String& matSymbol, G4bool create )
{
    fStart = "\n" ;
    fBefItem  = "\t\t\t\t" ;
    fAftItem  = "\n" ;
    fEnd   = "" ;


    G4Polyhedron* pPolyhedron ;

    //  visualization/management/src/G4VSceneHandler.cc

    G4int noofsides = 24 ; 
    G4Polyhedron::SetNumberOfRotationSteps (noofsides);
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());
       if( create ){     
           AddMeta( "create", "1" );
           pPolyhedron = solid->CreatePolyhedron ();  // always create a new poly   
       } else {
           AddMeta( "create", "0" );
           pPolyhedron = solid->GetPolyhedron ();     // if poly created already and no parameter change just provide that one 
       }
    }

    AddMeta( "cout", coutbuf.str() );
    AddMeta( "cerr", cerrbuf.str() );

    G4Polyhedron::ResetNumberOfRotationSteps ();
    //pPolyhedron -> SetVisAttributes (fpVisAttribs);

    const G4Polyhedron& polyhedron = *pPolyhedron ; 

    std::string polysmry ; 
    {
       std::stringstream ss ; 
       ss << "v " << polyhedron.GetNoVertices() << " " ; 
       ss << "f " << polyhedron.GetNoFacets() << " " ; 
       polysmry = ss.str();
    }
    AddMeta( "polysmry", polysmry );

    SetNoVertices(polyhedron.GetNoVertices());
    SetNoFacets(polyhedron.GetNoFacets());
    SetNoTexels(1);

    Metadata(polyhedron);   
    Vertices(polyhedron);   
    Normals(polyhedron);   
    Texels(matSymbol);

}

std::string G4DAEPolyhedron::IntAsString( G4int val  )
{
    std::ostringstream ss ;
    ss << val ; 
    return ss.str();
}

void G4DAEPolyhedron::AddMeta( const std::string& key, const std::string& val )
{
    fMetadata.insert(KV(key,val));
}

void G4DAEPolyhedron::Metadata( const G4Polyhedron& polyhedron )
{
    //
    // G4Polyhedron isa : HepPolyhedron, G4Visible
    //
    fMetadata.insert(KV("NumberOfRotationStepsAtTimeOfCreation", IntAsString(polyhedron.GetNumberOfRotationStepsAtTimeOfCreation())));
    fMetadata.insert(KV("NumberOfRotationSteps",                 IntAsString(polyhedron.GetNumberOfRotationSteps())));
#ifdef _GEANT4_TMP_GEANT94_
    G4int iebp = 1;
#else
    G4int iebp = polyhedron.IsErrorBooleanProcess() ? 1 : 0 ; 
#endif
    fMetadata.insert(KV("ErrorBooleanProcess",                   IntAsString(iebp)));

    const G4VisAttributes* visAtt = polyhedron.GetVisAttributes();
    std::ostringstream ss ;
    ss << *visAtt << G4endl ; 
    //G4cout << "visAtt " << *visAtt << G4endl; // SCB
    fMetadata.insert(KV("VisAttributes", ss.str()));
}

void G4DAEPolyhedron::Texels( const G4String& /*matSymbol*/ )
{
    std::ostringstream ss ;
    ss << fStart ; 
    {
        ss << fBefItem ; 
        ss << "1.0" << " " ;
        ss << "1.0" << " " ;
        ss << fAftItem ; 
    }
    ss << fEnd ; 
    fTexels = ss.str();
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

    G4int itexl = 1 ; // simplifying assumption of a single texel for entire polyhedron

    for (iface = 1, jface = nface; jface; jface--, iface++ ) 
    {
        norm = polyhedron.GetUnitNormal(iface);  
        ss << fBefItem  ; 
        ss << norm.x() << " " ; 
        ss << norm.y() << " " ; 
        ss << norm.z() << " " ; 
        ss << fAftItem  ; 

        Facet( polyhedron, iface , itexl ) ;
    }          
    ss << fEnd ; 
    fNormals = ss.str() ;
}


void G4DAEPolyhedron::Facet( const G4Polyhedron& polyhedron, G4int iface, G4int itexl )
{
    /*
      defines single collection of indices of the  "p" element 
    */
    G4int nedge;
    G4int ivertex[4];
    G4int iedgeflag[4];
    G4int ifacet[4];
    polyhedron.GetFacet(iface, nedge, ivertex, iedgeflag, ifacet);
    std::ostringstream ss ;
    std::ostringstream vv ;
    vv << nedge << " " ;

    G4int iedge = 0;
    for(iedge = 0; iedge < nedge; ++iedge) {
        // collada expects zero based indices
        ss << ivertex[iedge] - 1 << " " << iface - 1 << " " << itexl - 1 << "  " ;  
    }
    ss << " " ;
    std::string facet = ss.str() ; 
    fFacets.push_back( facet );

    std::string vcount = vv.str() ; 
    fVcount.push_back( vcount );
}




