#include "DsOpStackAction.h"

#include "DetDesc/IPVolume.h"
#include "DetHelpers/ICoordSysSvc.h"
#include "DetDesc/DetectorElement.h"
#include "DetDesc/IGeometryInfo.h"
#include "GaudiKernel/Service.h"
#include "GaudiKernel/DeclareFactoryEntries.h"

#include <G4ClassificationOfNewTrack.hh>
#include <G4SDManager.hh>
#include <G4RunManager.hh>
#include <G4TrackStatus.hh>
#include <G4ParticleDefinition.hh>
#include <G4Event.hh>
#include <G4ParticleTypes.hh>
#include <G4Track.hh>

DECLARE_TOOL_FACTORY(DsOpStackAction);


DsOpStackAction::DsOpStackAction ( const std::string& type   , 
				   const std::string& name   , 
				   const IInterface*  parent ) 
  : GiGaStackActionBase( type, name, parent ) ,
    stage(0),
    PhotonNumbers(0),
    NeutronNumbers(0),
    interestingEvt(false),
    m_csvc(0)
{ 
    declareProperty("TightCut",m_tightCut = false, " cut to select Neutron only event in the AD.");
    declareProperty("PhotonCut",m_photonCut = false, " Kill all the optical photons in the process.");
    declareProperty("MaxPhoton",m_maxPhoton = 1e6, " Max number of photons to be hold.");
}


StatusCode DsOpStackAction::initialize() 
{
  info() << "DsOpStackAction::initialize()" << endreq;
  
  StatusCode sc = GiGaStackActionBase::initialize();
  if (sc.isFailure()) return sc;

 
  if ( service("CoordSysSvc", m_csvc).isFailure()) {
    error() << " No CoordSysSvc available." << endreq;
    return StatusCode::FAILURE;
  }
  
  return StatusCode::SUCCESS; 
}

StatusCode DsOpStackAction::finalize() 
{
  info() << "DsOpStackAction::finalize()" << endreq;
  neutronList.clear();
  return  GiGaStackActionBase::finalize();
}

//--------------------------------------------------------------------------

G4ClassificationOfNewTrack DsOpStackAction::ClassifyNewTrack (const G4Track* aTrack) {
  
  //  info() << "DsOpStackAction::ClassifyNewTrack: " << endreq;
  
  G4ClassificationOfNewTrack classification = fUrgent;
  //info()<< " ParentID: "<< aTrack->GetParentID()<< " CurrentID: "<<aTrack->GetTrackID()<<endreq;
  

  switch(stage)
    {
    case 0: // if optical photon is the the secondary particles and below memory threshold,
      // put them in waiting.
      
      // tightcut selection here.
      if(aTrack->GetDefinition()  != G4OpticalPhoton::OpticalPhotonDefinition()){
	if(m_tightCut){
	  if( aTrack->GetDefinition()==G4Neutron::NeutronDefinition())
	    {
	      info()<<" It is a neutron event! " <<endreq;
	      NeutronNumbers++;
	      neutronList.push_back(aTrack->GetTrackID()); //save neutron's trackID for later use.
	      break;
	    }
	  
	  
	  if( aTrack->GetDefinition()==G4Gamma::GammaDefinition()
	      && (aTrack->GetTrackID()-aTrack->GetParentID())==1)  //only if the gamma has a direct parent.
	    {
	      if(!interestingEvt){
		interestingEvt=this->IsAInterestingTrack(aTrack);
		//info()<< "Particle: "<<aTrack->GetDefinition()->GetParticleName() <<endreq;
	      }
	      break;	    
	    }
	}
	else {
	  if( aTrack->GetDefinition()==G4Neutron::NeutronDefinition())
	    {
	      NeutronNumbers++;
	      break;
	    }
	  if (aTrack->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition()) {
	    if(!interestingEvt){
	      interestingEvt=this->PossibleInterestingTrack(aTrack);
	    }
	    break;
	  }
	}
      }
      else{
	
	PhotonNumbers++;
	
	
	// ------unother way of doing this is to use the processname to select events--
	//  if(aTrack->GetCreatorProcess()){
	//    G4String ProcessName=aTrack->GetCreatorProcess()->GetProcessName();
	//    info()<< " Proccess Name: "<< ProcessName<<endreq;
	//  }
	
	
	//// if m_photonCut selected, don't propogate any photon
	if (m_photonCut) {
	  classification=fKill;
	  break;
	}
	else if(aTrack->GetParentID()>0 && PhotonNumbers<=m_maxPhoton && !interestingEvt)
	  {
	    //if too many optical photons been generated and  too many optical photons hold
	    //in the stack, may cause 'out of memory' problem
	    
	    // only hold optical photon if it is secondary
	    
	    //	G4ThreeVector position  = aTrack->GetPosition();
	    //	G4ThreeVector direction = aTrack->GetMomentumDirection();
	    //	G4ThreeVector polarization = aTrack->GetPolarization();
	    //	G4double energy = aTrack->GetKineticEnergy();
	    //	G4double time  = aTrack->GetGlobalTime();
	    //	info()<<  "  Position  " << position.x()<< " time: " <<time<<endreq;
	    classification=fWaiting;
	    break;
	  }
      }
      
      classification = fUrgent;
      break;
      
    case 1:
      //      info()<<" In Stage 1, propogating all the stacked optical photons! "<<endreq;
      classification = fUrgent;
      break;
      
    default:
      classification = fUrgent;
    } 
  return classification;
}

//-------------------------------------------------------------------------
//-------------throw away non-interesting events, and-----------------------
//------------propogate optical photons otherwise ------------------
//-------------------------------------------------------------------

void DsOpStackAction::NewStage()
{

  info()<< " StackingAction::NewStage! "<<endreq;
  info()<< " Number of Optical Photons generated:  "<< PhotonNumbers<<endreq;
  info()<< " Number of Neutron generated:  "<< NeutronNumbers<<endreq;
  
  if(m_tightCut){
    info() << "Tight Cut selected: only select AD gamma events from neutrons! " <<endreq;
  }
  else{
    info() << " select Events with any new particles generated in the AD! " <<endreq;
  }

  if(PhotonNumbers>m_maxPhoton) {
    info() << " Get an event with Large number of Photons! " <<endreq;
  }
  
  if(m_photonCut){
    info() << " All the Optical Photons killed in this event! " <<endreq;
  }

  stage++;
  
  if(interestingEvt)
    {
      info()<<" An interesting event! Let's go on!"<<endreq;
      stackManager->ReClassify();
    }
  else {
    info()<< "Boring event, aborting..."<<endreq;
    stackManager->clear();  //abort the event
  }
}


//----------------------Reset -----------------------------------
void DsOpStackAction::PrepareNewEvent()
{
  info()<< " StackingAction::PrepareNewEvent! "<<endreq;
  interestingEvt=false;
  stage=0;
  PhotonNumbers=0;
  NeutronNumbers=0;
  neutronList.clear();
}

//-----------------If the Gamma neutron's daughter ? ---------------------

G4bool DsOpStackAction::IsNeutronDaughter(const G4int id, const vector<G4int> aList)
{
  
  //check if the gamma is the daughter of neutrons.
  G4bool isDaughter(false);
  for(size_t ii=0;ii<aList.size();ii++){
    if(aList[ii]==id || aList[ii]==id-1) {
      info()<<" neutron TrackID: "<<aList[ii]<< " been Captured! "<< endreq;
      isDaughter=true;
      break;
    }
  }
  return isDaughter;
};

// ------------- If the neutron been captured in the AD ? -------------------

G4bool DsOpStackAction::IsAInterestingTrack(const G4Track* aTrack)
{
  
  //  info()<< " Am I an interesting event???" <<endreq;
  
  G4int trkID=aTrack->GetParentID();  //get Gamma's parentID
  
  IDetectorElement *de;
  Gaudi::XYZPoint gp(aTrack->GetPosition().x(),aTrack->GetPosition().y(),aTrack->GetPosition().z());
  
  //Get DetectorElement from the global postion.
  de = m_csvc->coordSysDE(gp);
  if(!de){
    debug()<<" Particle Name: "<<aTrack->GetDefinition()->GetParticleName()<< " at position: "<<gp
	   <<" with Process Name: "<<aTrack->GetCreatorProcess()->GetProcessName()<<endreq;
  }
  
  //Ignore the track outside of our global volumes.
  if(de){
    IGeometryInfo *ginfo=de->geometry();
    if(ginfo){
      if( IsNeutronDaughter(trkID, neutronList))
	//if Gamma's parent is neutron, check if they are in the AD.
	{
	  //      G4String ProcessName=aTrack->GetCreatorProcess()->GetProcessName();
	  const ILVolume *lv=ginfo->lvolume();
	  if(lv){
	    G4String MaterialName = lv->materialName();
	    
	    //	    info() << " materialName: "<< MaterialName <<endreq;
	    
	    if( MaterialName=="/dd/Materials/MineralOil"
		|| MaterialName== "/dd/Materials/GdDopedLS"
		|| MaterialName== "/dd/Materials/LiquidScintillator" 
		|| MaterialName== "/dd/Materials/Acrylic") {
	      
	      info()<< "Find a Interesting Event in %s !!!" << MaterialName<<endreq; 
	      return true;
	    }
	  }
	}
    }
  }
  return false;
}


// ------------- If any new particles generated in the AD --------------------

G4bool DsOpStackAction::PossibleInterestingTrack(const G4Track* aTrack)
{
  
  //info()<< " Am I an possible interesting event???" <<endreq;
  
  IDetectorElement *de;
  Gaudi::XYZPoint gp(aTrack->GetPosition().x(),aTrack->GetPosition().y(),aTrack->GetPosition().z());
  
  //Get DetectorElement from the global postion.
  de = m_csvc->coordSysDE(gp);
  
  // If the new particle generated inside of the AD, accept it
  if(de){
    IGeometryInfo *ginfo=de->geometry();
    if(ginfo){
      {
	const ILVolume *lv=ginfo->lvolume();
	if(lv){
	  G4String MaterialName = lv->materialName();
	  
	  if( MaterialName=="/dd/Materials/MineralOil"
	      || MaterialName== "/dd/Materials/GdDopedLS"
	      || MaterialName== "/dd/Materials/LiquidScintillator" 
	      || MaterialName== "/dd/Materials/Acrylic") {
	    
	    info()<< "Find a good Event in AD in "<<  MaterialName<< " !! "<< endreq; 
	    return true;
	  }
	}
      }
    }
  }
  return false;
}



//-----------------------END----------------------------------
