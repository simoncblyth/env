#ifndef SENSDET_H
#define SENSDET_H

#include "G4VSensitiveDetector.hh"
#include "G4DataHelpers/G4DhHit.h"

#include <map>

class SensDet : public G4VSensitiveDetector {

public:
    SensDet(const std::string& name);
    virtual ~SensDet();

    int initialize() ; 
    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step,
                             G4TouchableHistory* history);


private:
    typedef std::map<short int,G4DhHitCollection*> LocalHitCache;
    LocalHitCache m_hc;


};

#endif

