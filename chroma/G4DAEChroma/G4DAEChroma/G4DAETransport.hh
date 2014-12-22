#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class G4DAESocketBase ;

#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEFotonList.hh"
#include "G4DAEChroma/G4DAEXotonList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"

class G4DAEMetadata ; 


class G4DAETransport
{
public:
    G4DAETransport(const char* envvar);
public:
    virtual ~G4DAETransport();

    void ClearAll();
    void Handshake(G4DAEMetadata* request=NULL);
    G4DAEMetadata* GetHandshake();

    void CollectPhoton(const G4Track* aPhoton );
    void CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid=-1);
    void DumpPhotons(bool /*hit*/) const ;
    void GetPhoton( std::size_t index , G4ThreeVector& pos, G4ThreeVector& dir, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const ;


    std::size_t ProcessCerenkovSteps(int batch_id);
    std::size_t ProcessScintillationSteps(int batch_id);
    std::size_t Propagate(int batch_id);
    std::size_t Process(int batch_id, G4DAEArrayHolder* request);


    // not yet using templated DAEList 
    G4DAEPhotonList* GetPhotons();
    G4DAEPhotonList* GetHits();
    void SetPhotons(G4DAEPhotonList* photons);
    void SetHits(   G4DAEPhotonList* hits);


    G4DAECerenkovStepList* GetCerenkovStepList();
    G4DAEScintillationStepList* GetScintillationStepList();
    G4DAEFotonList* GetFotonList();
    G4DAEXotonList* GetXotonList();

    // setters take ownership of photons/hits
    void SetCerenkovStepList(G4DAECerenkovStepList* cerenkov);
    void SetScintillationStepList(G4DAEScintillationStepList* scintillation);
    void SetFotonList(G4DAEFotonList* fotons);
    void SetXotonList(G4DAEXotonList* xotons);


    int GetVerbosity();
    void SetVerbosity(int verbosity);
 
private:

    G4DAESocketBase* m_socket ;

    G4DAEPhotonList* m_photons ; 

    G4DAEPhotonList* m_hits  ; 

    G4DAECerenkovStepList* m_cerenkov  ; 

    G4DAEScintillationStepList* m_scintillation  ; 

    G4DAEFotonList* m_fotons  ; 

    G4DAEXotonList* m_xotons  ; 

    G4DAEMetadata* m_handshake ; 

    int m_verbosity ; 

};

#endif 
