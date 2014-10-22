#ifndef G4DAEDAYABAY_H
#define G4DAEDAYABAY_H 1

#include "G4DAEChroma/G4DAEDetector.hh"
#include <map>

#include "G4DataHelpers/G4DhHit.h"

class G4DAEDayabay : public G4DAEDetector {

public:
    G4DAEDayabay();
    virtual ~G4DAEDayabay();

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



