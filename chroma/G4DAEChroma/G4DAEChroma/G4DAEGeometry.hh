#ifndef G4DAEGEOMETRY_H
#define G4DAEGEOMETRY_H 1

#include <cstddef>
#include <vector>
#include <string>
#include <map>

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"
#include "G4TouchableHistory.hh"
#include "geomdefs.hh"

class G4VPhysicalVolume ; 
class G4LogicalVolume ;
class G4VSensitiveDetector ; 
class G4DAETransformCache ;


/*
   tis unfocussed => split 

   * GDML loading and SD faking  
   * transform cache

    hmm the cache has a life of its own, and for 
    standalone use is all that is needed : so maybe
    rethink to make the cache a product of the geometry
    rather than a constituent 


*/




class G4DAEGeometry 
{
public:
    typedef std::map<std::size_t,std::size_t> PVSDMap_t ;
    typedef std::vector<G4VPhysicalVolume*> PVStack_t;

    static G4DAEGeometry* MakeGeometry( const char* geometry );
    static G4DAEGeometry* LoadFromGDML( const char* geokey="DAE_NAME_DYB_GDML", G4VSensitiveDetector* sd=NULL);
    static G4DAEGeometry* Load( const G4VPhysicalVolume* world=NULL );

    G4DAEGeometry();
    virtual ~G4DAEGeometry();
    void Clear();

public:
    // GDML looses SD assignment, so Fake these based on LV name matching 
    bool VisitFakeAssignSensitive(G4LogicalVolume* lv, PVStack_t pvStack, G4VSensitiveDetector* sd);
    void FakeAssignSensitive(G4LogicalVolume* lv, PVStack_t pvStack, G4VSensitiveDetector* sd);
  
    void AddSensitiveLVName(const std::string& lvname);
    void AddSensitiveLVNames(const char* envkey, char delim=';' );
    void DumpSensitiveLVNames();

public:
    G4DAETransformCache* CreateTransformCache(const G4VPhysicalVolume* world=NULL);

public:
    // default implementation returns the 1 based SD count (m_sdcount + 1) value 
    // expected to be overridden in detector specialization subclasses, 
    // implementation expected to return zero when no identifier is associated

    virtual std::size_t TouchableToIdentifier( const G4TouchableHistory& hist );

public:
    // move to another class ?
    void Dump();
    G4AffineTransform* GetNodeTransform(std::size_t index);
    std::string& GetNodeName(std::size_t index);

 
protected:
    void TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack, G4DAETransformCache* cache );
    void VisitPV(const G4LogicalVolume* const volumePtr, const PVStack_t pvStack, G4DAETransformCache* cache);
    EVolume VolumeType(G4VPhysicalVolume* pv) const;

private:

    // used by FakeAssignSensitive()  
    std::vector<std::string> m_lvsensitive;   
    std::vector<std::string> m_pvname;   // for debug, not identity matching 
    std::vector<G4AffineTransform> m_transform ; 

    std::size_t m_pvcount ; 
    std::size_t m_sdcount ; 

    PVSDMap_t m_pvsd ; 


};

#endif

