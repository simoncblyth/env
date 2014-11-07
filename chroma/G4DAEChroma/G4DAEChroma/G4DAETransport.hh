#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>
#include "G4ThreeVector.hh"

class G4Track ; 
class ZMQRoot ; 
class ChromaPhotonList ;

class G4DAETransport
{
public:
    static G4DAETransport* MakeTransport(const char* envvar="G4DAECHROMA_CLIENT_CONFIG");
protected:
    G4DAETransport(const char* envvar);
public:
    virtual ~G4DAETransport();

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
