#ifndef G4DAEDAYABAY_H
#define G4DAEDAYABAY_H 1

#include "G4DAEChroma/G4DAEDetector.hh"
#include <map>

#include "G4DataHelpers/G4DhHit.h"

class G4DAEDayabay : public G4DAEDetector {

public:
    G4DAEDayabay();
    virtual ~G4DAEDayabay();

    void DefineCollectionNames();
    void CreateHitCollections( const char* sdname, G4HCofThisEvent* HCE );
    void CollectHit( const G4DAEHit& hit );

private:
    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
    LocalHitCache m_hc;

    G4CollectionNameVector collectionName ; 
 

 
};

#endif



