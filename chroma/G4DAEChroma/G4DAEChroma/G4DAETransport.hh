#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class G4DAESocketBase ;
class G4DAEPhotons ; 
class G4DAECerenkovStepList ; 
class G4DAEScintillationStepList ; 
class G4DAEMetadata ; 


class G4DAETransport
{
public:
    G4DAETransport(const char* envvar);
public:
    virtual ~G4DAETransport();

    void ClearAll();
    void Handshake(G4DAEMetadata* request=NULL);

    std::size_t Propagate(int batch_id);
    void CollectPhoton(const G4Track* aPhoton );
    void CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid=-1);

    G4DAEPhotons* GetPhotons();
    G4DAEPhotons* GetHits();
    G4DAECerenkovStepList* GetCerenkovStepList();
    G4DAEScintillationStepList* GetScintillationStepList();
    G4DAEMetadata* GetHandshake();

    // setters take ownership of photons/hits
    void SetPhotons(G4DAEPhotons* photons);
    void SetHits(   G4DAEPhotons* hits);
    void SetCerenkovStepList(G4DAECerenkovStepList* cerenkov);
    void SetScintillationStepList(G4DAEScintillationStepList* scintillation);
 
private:

    G4DAESocketBase* m_socket ;

    G4DAEPhotons* m_photons ; 

    G4DAEPhotons* m_hits  ; 

    G4DAECerenkovStepList* m_cerenkov  ; 

    G4DAEScintillationStepList* m_scintillation  ; 

    G4DAEMetadata* m_handshake ; 

};

#endif 
