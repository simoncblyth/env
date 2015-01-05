/*

Aiming for this class to present an unchanging 
interface, on top of the flexible internal machinery 
that can adopt different:

* transport mechanism
* hit collection 


*/
#ifndef G4DAECHROMA_H
#define G4DAECHROMA_H 1

#include <string>
#include <cstddef>
#include "G4ThreeVector.hh"

class G4HCofThisEvent ; 

class G4DAEGeometry ;
class G4DAETransport ;
class G4DAESensDet ;
class G4DAECollector ;
class G4DAETransformCache ;
class G4DAEDatabase;
class G4DAEMetadata;

#include "G4DAEChroma/G4DAEManager.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAEScintillationPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovPhotonList.hh"
#include "G4DAEChroma/G4DAEPmtHitList.hh"


class G4DAEMaterialMap;
class G4Track ; 
class G4Run ;
class G4Event ;

#ifdef DEBUG_HITLIST
class G4DAEHitList;
#endif

class G4DAEChroma  : public G4DAEManager 
{

public:
    static G4DAEChroma* GetG4DAEChroma();
    static G4DAEChroma* GetG4DAEChromaIfExists();
protected:
    G4DAEChroma();
public:
    virtual ~G4DAEChroma();

    void SetGeometry(G4DAEGeometry* geo);
    G4DAEGeometry* GetGeometry();

    void SetTransport(G4DAETransport* tra);
    G4DAETransport* GetTransport();

    // standalone SD and collector with own hit collections
    void SetSensDet(G4DAESensDet* sd);
    G4DAESensDet* GetSensDet();
    G4DAECollector* GetCollector();

    // SD and collector with stolen pointers to preexisting hit collections
    void SetTrojanSensDet(G4DAESensDet* sd);
    G4DAESensDet* GetTrojanSensDet();
    G4DAECollector* GetTrojanCollector();

    // pointer to one of the two above SD
    void SetActiveSensDet(G4DAESensDet* sd);
    G4DAESensDet* GetActiveSensDet();
    G4DAECollector* GetActiveCollector();


    void SetTransformCache(G4DAETransformCache* cache);
    G4DAETransformCache* GetTransformCache();

    void SetDatabase(G4DAEDatabase* database);
    G4DAEDatabase* GetDatabase();

    void SetMetadata(G4DAEMetadata* metadata);
    G4DAEMetadata* GetMetadata();

    void SetHCofThisEvent(G4HCofThisEvent* hce);
    G4HCofThisEvent* GetHCofThisEvent();

    void SetVerbosity(int verbosity);
    int GetVerbosity();

    void SetControlId(int cid);
    int GetControlId();


    void Print(const char* msg="G4DAEChroma::Print");

    G4DAECerenkovStepList* GetCerenkovStepList();

    G4DAEScintillationStepList* GetScintillationStepList();

    G4DAEScintillationPhotonList* GetScintillationPhotonList();

    G4DAECerenkovPhotonList* GetCerenkovPhotonList();

    G4DAEPmtHitList* GetPmtHitList();


#ifdef DEBUG_HITLIST
    // from the SensDet collector
    G4DAEHitList* GetHitList();
#endif

public:
    void Handshake(G4DAEMetadata* request=NULL);

    G4DAEMetadata* GetHandshake();

    void SetMaterialMap(G4DAEMaterialMap* mmap);
    G4DAEMaterialMap* GetMaterialMap();

    void SetMaterialLookup(int* g2c);
    int* GetMaterialLookup();

    void SetPhotons(G4DAEPhotonList* photons);
    G4DAEPhotonList* GetPhotons();

    void SavePhotons(const char* evtkey );
    void LoadPhotons(const char* evtkey );

    void SetHits(G4DAEPhotonList* hits);
    G4DAEPhotonList* GetHits();

    void ClearAll();
    void CollectPhoton(const G4Track* aPhoton );

    std::size_t ProcessCerenkovSteps();
    std::size_t ProcessScintillationSteps();
    std::size_t ProcessSteps(G4DAEArrayHolder* steps);

    std::size_t ProcessCerenkovPhotons();
    std::size_t ProcessScintillationPhotons();
    std::size_t ProcessPhotons(G4DAEArrayHolder* photons);

    void AttachControlMetadata(G4DAEArrayHolder* request);
    std::size_t CollectHits(G4DAEArrayHolder* hits);

    void ChromaProcess(); 
    G4DAEPhotonList* GenerateMockPhotons();

public:
    void BeginOfEvent( const G4Event* event );
    void EndOfEvent( const G4Event* event );
    void BeginOfRun( const G4Run* run );
    void EndOfRun(   const G4Run* run );
    void Note( const char* msg );

 
private:
  // Singleton instance
  static G4DAEChroma* fG4DAEChroma;


private:

  // Photon Transport 
  G4DAETransport* m_transport ; 

  // Hit collection
  G4DAESensDet* m_sensdet ; 

  // Trojan
  G4DAESensDet* m_trojan_sensdet ; 

  // Active
  G4DAESensDet* m_active_sensdet ; 

  // Geometry Transform cache, used to convert global to local coordinates
  G4DAEGeometry* m_geometry ; 

  // Transform Cache  
  G4DAETransformCache* m_cache ; 

  // DB  
  G4DAEDatabase* m_database ; 

  // Metadata, separate from the photon and hits metadata
  G4DAEMetadata* m_metadata ; 

  // mapping between Geant4 and Chroma material indices
  G4DAEMaterialMap* m_materialmap ; 

  //
  G4HCofThisEvent* m_hce ;   

  // mapping between Geant4 and Chroma material indices
  int* m_g2c ; 

  // verbosity level
  int m_verbosity ;  

  // control id
  int m_cid ;  


};


#endif


