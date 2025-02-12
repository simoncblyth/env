#ifndef DSOPSTACKACTION_H
#define DSOPSTACKACTION_H 1

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

    ----- Kevin Zhang, Feb 2009
*/


class G4Track;
class IGeometryInfo;
class ICoordSysSvc;

using namespace std;

class DsOpStackAction :  public GiGaStackActionBase
{
  public:

  DsOpStackAction( const std::string& type ,  const std::string& name , const IInterface*  parent ) ;
  virtual ~DsOpStackAction() {};
  
  virtual StatusCode         initialize () ; 
  virtual StatusCode         finalize   () ;


  public:
    virtual G4ClassificationOfNewTrack ClassifyNewTrack( const G4Track* aTrack);
    virtual void NewStage();
    virtual void PrepareNewEvent();
    virtual G4bool    IsNeutronDaughter(const G4int id, const vector<G4int> aList);
    virtual G4bool    IsAInterestingTrack(const G4Track* aTrack);
    virtual G4bool    PossibleInterestingTrack(const G4Track* aTrack);
    
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
    
    //    IGeometryInfo* m_geo;
    
    // Locally cached pointer to the CoordSysSvc
    ICoordSysSvc* m_csvc;
    
    //    std::string              m_VertexSelection;
  
};

#endif

