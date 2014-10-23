#ifndef G4DAEDAYABAYCOLLECTOR_H
#define G4DAEDAYABAYCOLLECTOR_H 1

#include "G4DAEChroma/G4DAECollector.hh"
#include <map>

#include "G4DataHelpers/G4DhHit.h"

class DsChromaG4DAECollector : public G4DAECollector {

public:
    DsChromaG4DAECollector();
    virtual ~DsChromaG4DAECollector();

    void DefineCollectionNames(G4CollectionNameVector&);
    void CreateHitCollections( const char* sdname, G4HCofThisEvent* HCE );
    void StealHitCollections( const char* target,  G4HCofThisEvent* hce );

    void Collect( const G4DAEHit& hit );
    void AddSomeFakeHits();

private:
    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
    LocalHitCache m_hc;

    G4CollectionNameVector collectionName ; 
 

 
};

#endif



