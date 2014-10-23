#ifndef G4DAESENSDET_H
#define G4DAESENSDET_H

#include "G4VSensitiveDetector.hh"

class G4DAEGeometry ; 
class G4DAECollector ; 
class ChromaPhotonList ;

class G4DAESensDet : public G4VSensitiveDetector {

public:
    static G4DAESensDet* MakeSensDet(const char* name, const char* target=NULL);

    G4DAESensDet(const char* name, const char* target=NULL);
    virtual ~G4DAESensDet();

    int initialize() ; 
    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void SetCollector(G4DAECollector* col);
    G4DAECollector* GetCollector();

    void DumpStatistics( G4HCofThisEvent* HCE );

public:
    void CollectHits( ChromaPhotonList* cpl, G4DAEGeometry* geometry );
    void CollectOneHit( ChromaPhotonList* cpl , std::size_t index );

protected:
    G4DAECollector* m_collector ; 

private:
    G4DAEGeometry* m_geometry ; 
    const char* m_target ;

};

#endif

