

#ifndef TROJANSENSDET_H
#define TROJANSENSDET_H

#include "G4VSensitiveDetector.hh"
#include "G4DataHelpers/G4DhHit.h"

#include <map>
#include <string>

namespace DayaBay {
    class SimPmtHit;
}

// parasitically stores hits into target SD hit collections : eg DsPmtSensDet 
class TrojanSensDet : public G4VSensitiveDetector {

public:
    TrojanSensDet(const std::string& name, const std::string& target);
    virtual ~TrojanSensDet();

    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

public:
    std::string GetTargetName();
    G4VSensitiveDetector* GetTarget();
    void CheckTarget();
    void StoreHit(DayaBay::SimPmtHit* hit, int trackid);

private:
    void DumpStatistics( G4HCofThisEvent* HCE ) ; 
    int CacheHitCollections( const std::string& name, G4HCofThisEvent* HCE);

    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
    LocalHitCache m_hc;
    std::string m_target ; 

};

#endif


