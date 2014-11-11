#ifndef G4DAESENSDET_H
#define G4DAESENSDET_H

#include "G4VSensitiveDetector.hh"
#include <string>

class G4DAETransformCache ; 
class G4DAECollector ; 

#include "G4DAEChroma/Photons_t.hh"

class G4DAESensDet : public G4VSensitiveDetector {

public:
    static G4DAESensDet* MakeSensDet(const std::string& name, const std::string& target);

    G4DAESensDet(const std::string& name, const std::string& target);
    virtual ~G4DAESensDet();

    int initialize() ; 
    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void SetCollector(G4DAECollector* col);
    G4DAECollector* GetCollector();

    void Print();
    void DumpStatistics( G4HCofThisEvent* HCE );

public:
    void CollectHits(Photons_t* photons, G4DAETransformCache* cache );

protected:
    std::string m_target ;
    G4DAECollector* m_collector ; 

};

#endif

