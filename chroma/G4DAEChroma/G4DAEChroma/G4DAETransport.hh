#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class G4DAESocketBase ;
#include "G4DAEChroma/G4DAEPhotons.hh"

// generalizing to support the old ChromaPhotonList demands a Photons virtual base
// but thats kinda complicated due to ROOT TObject persisting, so put that 
// on hold : as aiming to get rid of ROOT anyhow

class G4DAETransport
{
public:
    G4DAETransport(const char* envvar);
public:
    virtual ~G4DAETransport();

    void ClearAll();
    std::size_t Propagate(int batch_id);
    void CollectPhoton(const G4Track* aPhoton );
    void CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid=-1);

    G4DAEPhotons* GetPhotons();
    G4DAEPhotons* GetHits();
 
private:

    G4DAESocketBase* m_socket ;

    G4DAEPhotons* m_photons ; 

    G4DAEPhotons* m_hits  ; 


};

#endif 
