#ifndef DEMOG4DAECOLLECTOR_H
#define DEMOG4DAECOLLECTOR_H 1

#include "G4DAEChroma/G4DAECollector.hh"


class DemoG4DAECollector : public G4DAECollector  {

public:

    DemoG4DAECollector(){};
    virtual ~DemoG4DAECollector(){};

    void DefineCollectionNames(G4CollectionNameVector&);;
    void CreateHitCollections( const char* sdname, G4HCofThisEvent* hce );
    void StealHitCollections( const char* target,  G4HCofThisEvent* hce );
    void Collect( const G4DAEHit& hit );
    void HarvestPmtHits();


};

#endif

