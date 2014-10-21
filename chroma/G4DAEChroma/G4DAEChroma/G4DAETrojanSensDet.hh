#ifndef G4DAETROJANSENSDET_H
#define G4DAETROJANSENSDET_H

#include "G4DAESensDet.hh"
class G4DAEGeometry ; 

#include <string>


// parasitically stores hits into target SD hit collections : eg DsPmtSensDet 
class G4DAETrojanSensDet : public G4DAESensDet {

public:
    static G4DAETrojanSensDet* MakeTrojanSensDet(const std::string& target, G4DAEGeometry* geo );
    static G4DAETrojanSensDet* GetTrojanSensDet(const std::string& target);
protected:
    G4DAETrojanSensDet(const std::string& name, const std::string& target);

public:
    virtual ~G4DAETrojanSensDet();

    virtual void Initialize( G4HCofThisEvent* HCE ) ; 
    virtual void EndOfEvent( G4HCofThisEvent* HCE ) ; 
    virtual bool ProcessHits(G4Step* step, G4TouchableHistory* history);

private:
    int CacheHitCollections( const std::string& name, G4HCofThisEvent* HCE);
    std::string m_target ; 

};

#endif


