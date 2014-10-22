#ifndef G4DAEDETECTOR_H
#define G4DAEDETECTOR_H 1

#include "G4CollectionNameVector.hh"
#include "G4DAEChroma/G4DAEHit.hh"

class G4HCofThisEvent ;
class ChromaPhotonList ; 
class G4DAEGeometry ;

class G4DAEDetector  {

public:
    G4DAEDetector(){};
    virtual ~G4DAEDetector(){};

    virtual void DefineCollectionNames(G4CollectionNameVector&) = 0;
    virtual void CreateHitCollections( const char* sdname, G4HCofThisEvent* hce ) = 0;
    virtual void StealHitCollections( const char* target,  G4HCofThisEvent* hce ) = 0;
    virtual void Collect( const G4DAEHit& hit ) = 0;
    virtual void AddSomeFakeHits() = 0;

    virtual void DumpStatistics( G4HCofThisEvent* hce );
    virtual void CollectHits( ChromaPhotonList* cpl, G4DAEGeometry* geometry );


};

#endif


