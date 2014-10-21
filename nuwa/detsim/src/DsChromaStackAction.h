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
#include "G4VSensitiveDetector.hh"
//#include "GiGa/GiGaSensDetBase.h"

/*  A user defined class to select interesting events in the stack action.


    Based On DsOpStackAction  ----- Kevin Zhang, Feb 2009
*/


#include "G4DataHelpers/G4DhHit.h"

#include <string>
#include <map>


class G4Track;

class G4Step;
class G4TouchableHistory;

class IGeometryInfo;
class ICoordSysSvc;

class G4DAEChroma;


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


    virtual G4bool    IsNeutronDaughter(const G4int id, const std::vector<G4int> aList);
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
    
    // local ptr to singleton instance
    G4DAEChroma* m_chroma ; 

  
};

#endif

