#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class G4DAESocketBase ;
class G4DAESerializablePhotons ;

typedef G4DAESerializablePhotons Photons_t ;


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

    Photons_t* GetPhotons();
    Photons_t* GetHits();
 
private:

    G4DAESocketBase* m_socket ;

    Photons_t* m_photons ; 

    Photons_t* m_hits  ; 


};

#endif 
