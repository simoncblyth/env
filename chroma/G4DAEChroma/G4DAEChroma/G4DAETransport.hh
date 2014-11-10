#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class G4DAESocketBase ;
class G4DAEPhotonList ; 



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

    G4DAEPhotonList* GetPhotons();
    G4DAEPhotonList* GetHits();
 
private:

    G4DAESocketBase* m_socket ;

    G4DAEPhotonList* m_photons ; 

    G4DAEPhotonList* m_hits  ; 


};

#endif 
