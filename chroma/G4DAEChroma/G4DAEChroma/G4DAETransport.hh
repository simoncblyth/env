#ifndef G4DAETRANSPORT_H
#define G4DAETRANSPORT_H 1

#include <cstddef>

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

 
