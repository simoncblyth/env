#ifndef DSCHROMASTACKACTION_H
#define DSCHROMASTACKACTION_H 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "G4UserStackingAction.hh"
#include "G4ThreeVector.hh"
#include "GaudiAlg/GaudiTool.h"
#include "G4OpticalPhoton.hh"
#include "G4Neutron.hh"
#include "GiGa/GiGaStackActionBase.h"

/*  A user defined class to select interesting events in the stack action.


    Based On DsOpStackAction  ----- Kevin Zhang, Feb 2009
*/


class G4Track;
class IGeometryInfo;
class ICoordSysSvc;

class ZMQRoot ; 
class ChromaPhotonList ;


using namespace std;

class DsChromaStackAction :  public GiGaStackActionBase
{
  public:

  DsChromaStackAction( const std::string& type ,  const std::string& name , const IInterface*  parent ) ;
  virtual ~DsChromaStackAction() {};
  
  virtual StatusCode         initialize () ; 
  virtual StatusCode         finalize   () ;


  public:
    virtual G4ClassificationOfNewTrack ClassifyNewTrack( const G4Track* aTrack);
    virtual void NewStage();
    virtual void PrepareNewEvent();

    virtual void CollectPhoton(const G4Track* aPhoton );

    virtual G4bool    IsNeutronDaughter(const G4int id, const vector<G4int> aList);
    virtual G4bool    IsRelevantNeutronDaughter(const G4Track* aTrack);
    virtual G4bool    IsRelevant(const G4Track* aTrack);
    
  private:

    G4int stage;
    G4int PhotonNumbers;
    G4int NeutronNumbers;
    G4bool interestingEvt;    //    Is this event a possible background event? 

    std::vector<G4int> neutronList;

    // background selection property.
    G4bool m_tightCut;
    
    // kill all the optical photons if True
    G4bool m_photonCut;

    // Maximum Number of optical photons been hold..
    G4double m_maxPhoton;

    // Modulo scale down photons collected
    G4int m_moduloPhoton;

    
    //    IGeometryInfo* m_geo;
    
    // Locally cached pointer to the CoordSysSvc
    ICoordSysSvc* m_csvc;
    
    //    std::string              m_VertexSelection;

    // ZeroMQ network socket utility 
    ZMQRoot* fZMQRoot ; 

    // transport ready TObject 
    ChromaPhotonList* fPhotonList ; 

    // test receiving object from remote zmq server
    ChromaPhotonList* fPhotonList2 ; 


  
};

#endif

