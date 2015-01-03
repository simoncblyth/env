#ifndef DYBG4DAECOLLECTOR_H
#define DYBG4DAECOLLECTOR_H 1

#include "G4DAEChroma/G4DAECollector.hh"
#include <map>
#include <string>

#include "G4DataHelpers/G4DhHit.h"

// 
// Dayabay specialization of G4DAECollector
// allowing use of G4DAEChroma external photon 
// propagation machinery 
//

class DybG4DAECollector : public G4DAECollector {

public:
    DybG4DAECollector();
    virtual ~DybG4DAECollector();

    void DefineCollectionNames(G4CollectionNameVector&);
    void CreateHitCollections( const std::string& sdname, G4HCofThisEvent* HCE );
    void StealHitCollections( const std::string& target,  G4HCofThisEvent* hce );

    void Collect( const G4DAEHit& hit );
    void AddSomeFakeHits();
    void DumpLocalHitCache();
    void DumpLocalHitCollection(G4DhHitCollection* hc);
    void FillPmtHitList();

private:
    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
    LocalHitCache m_hc;

    G4CollectionNameVector collectionName ; 
 

 
};

#endif



