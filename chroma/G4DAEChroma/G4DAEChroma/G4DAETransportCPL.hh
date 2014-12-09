#ifdef G4DAECHROMA_WITH_CPL

#ifndef G4DAETRANSPORTCPL_H
#define G4DAETRANSPORTCPL_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

/*
TODO: Eliminate this, when demonstrate can transport CPL via the G4DAESocketBase
*/


class G4Track ; 
class ZMQRoot ; 
class ChromaPhotonList ;

class G4DAETransportCPL
{
public:
    G4DAETransportCPL(const char* envvar);
public:
    virtual ~G4DAETransportCPL();

    void ClearAll();
    std::size_t Propagate(int batch_id);
    void CollectPhoton(const G4Track* aPhoton );
    void CollectPhoton(const G4ThreeVector& pos, const G4ThreeVector& dir, const G4ThreeVector& pol, const float time, const float wavelength, const int pmtid=-1);

    // pmtid ordinarilly -1, but useful for fake-transport test

    ChromaPhotonList* GetPhotons();
    ChromaPhotonList* GetHits();
 
private:
    // ZeroMQ network socket utility 
    ZMQRoot* fZMQRoot ; 

    // transport ready TObject 
    ChromaPhotonList* fPhotonList ; 

    // test receiving object from remote zmq server
    ChromaPhotonList* fPhotonHits ; 


};

#endif 
#endif 
