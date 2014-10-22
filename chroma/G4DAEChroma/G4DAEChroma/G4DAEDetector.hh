#ifndef G4DAEDETECTOR_H
#define G4DAEDETECTOR_H 1

#include "G4CollectionNameVector.hh"
#include "G4DAEChroma/G4DAEHit.hh"

class G4HCofThisEvent ;

class G4DAEDetector  {
    G4DAEDetector(){};
    virtual ~G4DAEDetector(){};

    virtual void DefineCollectionNames() = 0;
    virtual void CreateHitCollections( const char* sdname, G4HCofThisEvent* hce ) = 0;
    virtual void CollectHit( const G4DAEHit& hit ) = 0;
};

#endif


