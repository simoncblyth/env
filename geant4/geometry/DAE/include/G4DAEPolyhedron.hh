/*

face(1)= 1/2 4/3 3/4 2/5  
face(2)= 5/6 8/3 4/1 1/5   
face(3)= 8/6 7/4 3/1 4/2   
face(4)= 7/6 6/5 2/1 3/3   
face(5)= 6/6 5/2 1/1 2/4   
face(6)= 5/5 6/4 7/3 8/2   

Illustration of the vertices and face normals 
for a G4PolyhedronBox
                                               
  Z     Y                                       
   |   /                                       
   |  /                                        
   | /                                         
   *----- X                                     
                 V8  +------------+ V7          
                    /            /             
                   /  F6 (+Z)   /              
                  /            /               
              V5 +------------+ V6              
                                                  
   (-X) F2                *              F4 (+X)       
                  V4 +------------+ V3             
                    /            /                 
                   /  F1 (-Z)   /                  
                  /            /                   
             V1  +------------+  V2                
                                                  
*/

#include "G4Polyhedron.hh"
#include "G4Point3D.hh"
#include "G4Normal3D.hh"

#include <string>
#include <sstream>
#include <vector>
#include <map>


class G4DAEPolyhedron
{
  public:
    G4DAEPolyhedron(const G4Polyhedron& polyhedron);
   ~G4DAEPolyhedron(){};

    void AddMeta( const std::string& key, const std::string& val );
    void Metadata( const G4Polyhedron& polyhedron );
    void Vertices( const G4Polyhedron& polyhedron );
    void Normals(  const G4Polyhedron& polyhedron );
    void Facet(    const G4Polyhedron& polyhedron, G4int iface );
    void Dump();

    std::string IntAsString( G4int val );
    std::string GetVertices(){ return fVertices ; }
    std::string GetNormals(){ return fNormals ; }
    std::vector<std::string>& GetFacets(){ return fFacets ; }
    std::vector<std::string>& GetVcount(){ return fVcount ; }
    std::map<std::string,std::string>& GetMetadata(){ return fMetadata ; }
 
  private:

    std::string fStart ; 
    std::string fBefItem ; 
    std::string fAftItem ; 
    std::string fEnd ; 

    std::string fVertices ; 
    std::string fNormals ;
    std::vector<std::string> fFacets ;
    std::vector<std::string> fVcount ;
    std::map<std::string,std::string> fMetadata ;
};



