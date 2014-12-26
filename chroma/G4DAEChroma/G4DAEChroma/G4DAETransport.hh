#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>

class G4DAESocketBase ;

#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEScintillationPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"

class G4DAEArrayHolder ; 
class G4DAEMetadata ; 


class G4DAETransport
{
public:
    G4DAETransport(const char* envvar);
public:
    virtual ~G4DAETransport();

    void ClearAll();

    /////////////   remote callers

    void Handshake(G4DAEMetadata* request=NULL);

    std::size_t ProcessCerenkovSteps(int batch_id);
    std::size_t ProcessScintillationSteps(int batch_id);
    std::size_t ProcessCerenkovPhotons(int batch_id);
    std::size_t ProcessScintillationPhotons(int batch_id);

    std::size_t Propagate(int batch_id);
    std::size_t Process(int batch_id, G4DAEArrayHolder* request);

    G4DAEArrayHolder* ProcessRaw(int batch_id, G4DAEArrayHolder* request);

    //////////////  getters

    G4DAEMetadata* GetHandshake();
    G4DAEArrayHolder* GetResponse();

    G4DAEPhotonList* GetPhotons();
    G4DAEPhotonList* GetHits();
    G4DAEScintillationPhotonList*  GetScintillationPhotonList();
    G4DAECerenkovPhotonList*  GetCerenkovPhotonList();

    G4DAECerenkovStepList* GetCerenkovStepList();
    G4DAEScintillationStepList* GetScintillationStepList();

    //////////////  setters take ownership of photons/hits

    void SetResponse(G4DAEArrayHolder* response);

    void SetPhotons(G4DAEPhotonList* photons);
    void SetHits(   G4DAEPhotonList* hits);
    void SetScintillationPhotonList(G4DAEScintillationPhotonList* scintillation_photons);
    void SetCerenkovPhotonList(G4DAECerenkovPhotonList* cerenkov_photons);

    void SetCerenkovStepList(G4DAECerenkovStepList* cerenkov);
    void SetScintillationStepList(G4DAEScintillationStepList* scintillation);

    ////////////

    int GetVerbosity();
    void SetVerbosity(int verbosity);
 
private:

    G4DAESocketBase* m_socket ;

    G4DAEMetadata* m_handshake ; 

private:

    G4DAECerenkovStepList* m_cerenkov  ; 

    G4DAEScintillationStepList* m_scintillation  ; 

private:

    G4DAEPhotonList* m_photons ; 

    G4DAEScintillationPhotonList* m_scintillation_photons  ; 

    G4DAECerenkovPhotonList* m_cerenkov_photons  ; 

private:

    G4DAEPhotonList* m_hits ; 

private:

    int m_verbosity ; 

};

#endif 
