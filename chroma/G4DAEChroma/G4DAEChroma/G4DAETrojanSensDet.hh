#ifndef G4DAETROJANSENSDET_H
#define G4DAETROJANSENSDET_H

#include "G4DAESensDet.hh"

#include <string>


// parasitically stores hits into target SD hit collections : eg DsPmtSensDet 
class G4DAETrojanSensDet : public G4DAESensDet {

public:
    G4DAETrojanSensDet(const std::string& name, const std::string& target);
    virtual ~G4DAETrojanSensDet();

    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

public:
    std::string GetTargetName();
    G4VSensitiveDetector* GetTarget();
    void CheckTarget();

private:
    int CacheHitCollections( const std::string& name, G4HCofThisEvent* HCE);
    std::string m_target ; 

};

#endif


