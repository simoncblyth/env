#ifndef G4DAESENSDET_H
#define G4DAESENSDET_H

#include "G4VSensitiveDetector.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEPmtHitList.hh"
#include <string>

class G4DAETransformCache ; 
class G4DAECollector ; 

class G4DAESensDet : public G4VSensitiveDetector {

public:
    static G4DAESensDet* MakeTrojanSensDet(const std::string& target, G4DAECollector* collector);
    static G4DAESensDet* MakeChromaSensDet(const std::string& target, G4DAECollector* collector);

    G4DAESensDet(const std::string& name, const std::string& target);
    virtual ~G4DAESensDet();

    int initialize() ; 
    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

    void SetCollector(G4DAECollector* col);
    G4DAECollector* GetCollector();

    void PopulatePmtHitList(G4DAEPmtHitList* pmthits);

    void Print(const char* msg="G4DAESensDet::Print") const;
    void DumpStatistics( G4HCofThisEvent* HCE );

public:
    void CollectHits(G4DAEPhotonList* photons, G4DAETransformCache* cache );
    static void MockupSD(const char* name, G4DAECollector* collector);


protected:
    std::string m_target ;
    G4DAECollector* m_collector ; 

};

#endif

