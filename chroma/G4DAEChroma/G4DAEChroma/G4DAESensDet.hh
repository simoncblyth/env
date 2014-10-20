#ifndef G4DAESENSDET_H
#define G4DAESENSDET_H

#include "G4VSensitiveDetector.hh"
#include <map>

#ifdef G4DAE_DAYABAY
#include "G4DataHelpers/G4DhHit.h"
namespace DayaBay {
    class SimPmtHit;
}
#endif


class G4DAEGeometry ; 
class ChromaPhotonList ;

class G4DAESensDet : public G4VSensitiveDetector {

public:
    G4DAESensDet(const std::string& name);
    virtual ~G4DAESensDet();

    int initialize() ; 
    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void SetGeometry(G4DAEGeometry* geo);
    G4DAEGeometry* GetGeometry();

    void DumpStatistics( G4HCofThisEvent* HCE );
private:
    void CreateHitCollections( G4HCofThisEvent* HCE );
    void DefineCollectionNames();

public:
    void CollectHits( ChromaPhotonList* cpl );
    void CollectOneHit( ChromaPhotonList* cpl , std::size_t index );
    void AddSomeFakeHits();
#ifdef G4DAE_DAYABAY
    void StoreHit(DayaBay::SimPmtHit* hit, int trackid);
#else 
    void StoreHit(void* hit, int trackid);
#endif

protected:

#ifdef G4DAE_DAYABAY
    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
#else
    typedef std::map<short int,void*> LocalHitCache;
#endif
    LocalHitCache m_hc;

private:
    G4DAEGeometry* m_geometry ; 

};

#endif

