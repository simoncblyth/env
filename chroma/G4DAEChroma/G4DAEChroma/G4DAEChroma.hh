/*

G4DAEChroma
=============

Objective
------------

Pull out everything Chroma related and reusable 
from StackAction and SensitiveDetector
for flexible reusability in different Geant4 contexts

Dependencies
------------

* giga/gaudi/gauss NOT ALLOWED 
* sticking to plain Geant4, ZMQ, ZMQRoot,... for generality 

Issues
--------

Development Cycle too slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: Create test application for machinery test 
(enable to rapidly work on the marshalling) 

* reads Dyb geometry into G4 from exported GDML
* reads some initial photon positions from a .root file
* invokes this photon collection and propagation 
* dumps the hits returned


GPU Hit handling
~~~~~~~~~~~~~~~~~

* how to register DsChromaPmtSensDet instead of (or in addition to) DsPmtSensDet
  or some how get access to DsPmtSensDet

  * class name "DsPmtSensDet" is mentioned in DetDesc 
    logvol sensdet attribute, somehow DetDesc/GiGa 
    hands that over to Geant4 : need to swizzle OR add ? 

  * old approach duplicated bits of "DsPmtSensDet" for adding 
    hits into the StackAction : that was too messy then, but perhaps
    clean enough now have pulled out Chroma parts into G4DAEChroma 

  * but needs access to private methods from DsPmtSensDet, so 
    maybe a no-no anyhow : especially as need very little
    functionality 

* how to get access to DsPmtSensDet in order to add hits

  * provide singleton accessor for cheat access to globally 
    shared instance ? 
    Approach has MT complications : but no need to worry about that yet

  * gaudi has a way of accessing the instance, do it externally (where?)
    and pass it in 

* how to handle hits interfacing to detector specific code


*/
#ifndef G4DAECHROMA_H
#define G4DAECHROMA_H 1

#include <cstddef>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4AffineTransform.hh"

class ZMQRoot ; 
class ChromaPhotonList ;
class G4VPhysicalVolume ; 
class G4LogicalVolume ;
 
class G4DAEChroma 
{
public:
    typedef std::vector<G4VPhysicalVolume*> PVStack_t;

    struct Hit {
        // global
        G4ThreeVector gpos ;
        G4ThreeVector gdir ;
        G4ThreeVector gpol ;

       // local : maybe just keep local, inplace transform ?
        G4ThreeVector lpos ;
        G4ThreeVector ldir ;
        G4ThreeVector lpol ;
        float t ;
        float wavelength ;
        int   pmtid ;
        int   volumeindex ;

        void LocalTransform(G4AffineTransform& trans)
        { 
            lpos = trans.TransformPoint(gpos);
            lpol = trans.TransformAxis(gpol);
            lpol = lpol.unit();
            ldir = trans.TransformAxis(gdir);
            ldir = ldir.unit();
        }

    }; 

public:
    static G4DAEChroma* GetG4DAEChroma();
    static G4DAEChroma* GetG4DAEChromaIfExists();
protected:
    G4DAEChroma();
public:
    virtual ~G4DAEChroma();

    void ClearAll();
    void CollectPhoton(const G4Track* aPhoton );
    void Propagate(G4int batch_id);
    bool ProcessHitsChroma( const ChromaPhotonList* cpl, std::size_t index );

public:
    void CreateTransformCache(G4VPhysicalVolume* wpv=NULL);
    void DumpTransformCache();
    void TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack);
    void VisitPV( const PVStack_t& pvStack );
    EVolume VolumeType(G4VPhysicalVolume* pv) const;

    G4AffineTransform& GetNodeTransform(std::size_t index);
    std::string& GetNodeName(std::size_t index);

 
private:
  // Singleton instance
  static G4DAEChroma* fG4DAEChroma;

  // ZeroMQ network socket utility 
  ZMQRoot* fZMQRoot ; 

  // transport ready TObject 
  ChromaPhotonList* fPhotonList ; 

  // test receiving object from remote zmq server
  ChromaPhotonList* fPhotonList2 ; 

private:
  std::vector<std::string> m_pvname;   // for debug, not identity matching 
  std::vector<G4AffineTransform> m_transform ; 
  bool m_transform_cache_created ; 

};


#endif

 
