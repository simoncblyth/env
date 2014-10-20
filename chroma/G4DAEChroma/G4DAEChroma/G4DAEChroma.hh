/*
*/
#ifndef G4DAECHROMA_H
#define G4DAECHROMA_H 1

#include <cstddef>
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4AffineTransform.hh"

class ZMQRoot ; 
class ChromaPhotonList ;
class G4DAEGeometry ;
class G4DAETrojanSensDet ;

class G4DAEChroma 
{
public:
    static G4DAEChroma* GetG4DAEChroma();
    static G4DAEChroma* GetG4DAEChromaIfExists();
protected:
    G4DAEChroma(const char* envvar="G4DAECHROMA_CLIENT_CONFIG");
public:
    virtual ~G4DAEChroma();

    void SetGeometry(G4DAEGeometry* geo);
    G4DAEGeometry* GetGeometry();

    // **RegisterTrojanSD**
    //
    //     creates parasitic SD, re-registering with the same target does nothing  
    //     internally allows adding hits to hitcollections of target SD, 
    //     eg canonically DsPmtSensDet
    //
    //
    void RegisterTrojanSD(const std::string& target);

    void ClearAll();
    void CollectPhoton(const G4Track* aPhoton );
    void Propagate(G4int batch_id, const std::string& target);

    G4DAETrojanSensDet* GetTrojanSD(const std::string& target);
//private:
//    G4DAETrojanSensDet* GetTrojanSD(const std::string& target);
 
private:
  // Singleton instance
  static G4DAEChroma* fG4DAEChroma;

  // ZeroMQ network socket utility 
  ZMQRoot* fZMQRoot ; 

  // transport ready TObject 
  ChromaPhotonList* fPhotonList ; 

  // test receiving object from remote zmq server
  ChromaPhotonList* fPhotonList2 ; 

  // Geometry Transform cache, used to convert global to local coordinates
  G4DAEGeometry* fGeometry ; 

};


#endif

 
