#ifndef G4DAEGEOMETRY_H
#define G4DAEGEOMETRY_H 1

#include <cstddef>
#include <vector>
#include <map>

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"
#include "G4TouchableHistory.hh"
#include "geomdefs.hh"

class G4VPhysicalVolume ; 
class G4LogicalVolume ;
class G4VSensitiveDetector ; 

/*
   tis unfocussed => split 

   * GDML loading and SD faking  
   * transform cache

*/


class G4DAEGeometry 
{
public:
    typedef std::map<std::size_t,std::size_t> PVSDMap_t ;
    typedef std::map<std::size_t,G4AffineTransform> TransformMap_t ;
    typedef std::vector<G4VPhysicalVolume*> PVStack_t;

    static G4DAEGeometry* MakeGeometry( const char* geometry );
    static G4DAEGeometry* LoadFromGDML( const char* geokey="DAE_NAME_DYB_GDML", G4VSensitiveDetector* sd=NULL);
    static G4DAEGeometry* Load( const G4VPhysicalVolume* world=NULL );

    G4DAEGeometry();
    void Clear();
    virtual ~G4DAEGeometry();

public:
    bool CacheExists();


public:
    // GDML looses SD assignment, so Fake these based on LV name matching 
    bool VisitFakeAssignSensitive(G4LogicalVolume* lv, PVStack_t pvStack, G4VSensitiveDetector* sd);
    void FakeAssignSensitive(G4LogicalVolume* lv, PVStack_t pvStack, G4VSensitiveDetector* sd);
  
    void AddSensitiveLVName(const std::string& lvname);
    void AddSensitiveLVNames(const char* envkey, char delim=';' );
    void DumpSensitiveLVNames();

public:

    void CreateTransformCache(const G4VPhysicalVolume* wpv=NULL);
    void DumpTransformCache();

    G4AffineTransform* GetSensorTransform(std::size_t id);
    //std::string& GetSensorPVName(std::size_t id);

    G4AffineTransform* GetNodeTransform(std::size_t index);
    std::string& GetNodeName(std::size_t index);

// merging in IDMAP functionality from GaussTools GiGaRunActionExport

public:
    // default implementation returns the 1 based SD count (m_sdcount + 1) value 
    // expected to be overridden in detector specialization subclasses, 
    // implementation expected to return zero when no identifier is associated

    virtual std::size_t TouchableToIdentifier( const G4TouchableHistory& hist );

 
protected:
    void TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack);
    void VisitPV(const G4LogicalVolume* const volumePtr, const PVStack_t pvStack);
    EVolume VolumeType(G4VPhysicalVolume* pv) const;

private:

    // used by FakeAssignSensitive()  
    std::vector<std::string> m_lvsensitive;   


    std::vector<std::string> m_pvname;   // for debug, not identity matching 
    std::vector<G4AffineTransform> m_transform ; 
    bool m_transform_cache_created ; 

    std::size_t m_pvcount ; 
    std::size_t m_sdcount ; 
    PVSDMap_t m_pvsd ; 

    TransformMap_t m_id2transform ; 



};

#endif

