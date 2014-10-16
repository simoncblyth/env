#ifndef G4DAEGEOMETRY_H
#define G4DAEGEOMETRY_H 1

#include <cstddef>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"
#include "geomdefs.hh"

class G4VPhysicalVolume ; 
class G4LogicalVolume ;

class G4DAEGeometry 
{
public:
    static G4DAEGeometry* LoadFromGDML( const char* geokey="DAE_NAME_DYB_GDML" );

    G4DAEGeometry();
    virtual ~G4DAEGeometry();

public:
    bool CacheExists();
    void CreateTransformCache(G4VPhysicalVolume* wpv=NULL);
    void DumpTransformCache();
    G4AffineTransform& GetNodeTransform(std::size_t index);
    std::string& GetNodeName(std::size_t index);
 
protected:
    typedef std::vector<G4VPhysicalVolume*> PVStack_t;
    void TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack);
    void VisitPV( const PVStack_t& pvStack );
    EVolume VolumeType(G4VPhysicalVolume* pv) const;

private:
    std::vector<std::string> m_pvname;   // for debug, not identity matching 
    std::vector<G4AffineTransform> m_transform ; 
    bool m_transform_cache_created ; 

};

#endif

