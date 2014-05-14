#include "DsChromaStackAction.h"

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

#ifdef WITH_CHROMA_ZMQ
#include "ChromaPhotonList.hh"
#include "ZMQRoot.hh"
#endif


DECLARE_TOOL_FACTORY(DsChromaStackAction);

DsChromaStackAction::DsChromaStackAction ( const std::string& type   , 
				   const std::string& name   , 
				   const IInterface*  parent ) 
  : GiGaStackActionBase( type, name, parent ) ,
    stage(0),
    PhotonNumbers(0),
    NeutronNumbers(0),
    interestingEvt(false),
    m_csvc(0),
    fZMQRoot(0),
    fPhotonList(0),
    fPhotonList2(0)
{ 
    declareProperty("TightCut",m_tightCut = false, " cut to select Neutron only event in the AD.");
    declareProperty("PhotonCut",m_photonCut = false, " Kill all the optical photons in the process.");
    declareProperty("MaxPhoton",m_maxPhoton = 1e6, " Max number of photons to be hold.");
    declareProperty("ModuloPhoton",m_moduloPhoton = 100, "Modulo scale down photons collected.");
}


StatusCode DsChromaStackAction::initialize() 
{
  info() << "DsChromaStackAction::initialize()" << endreq;
  
  StatusCode sc = GiGaStackActionBase::initialize();
  if (sc.isFailure()) return sc;


  if ( service("CoordSysSvc", m_csvc).isFailure()) {
    error() << " No CoordSysSvc available." << endreq;
    return StatusCode::FAILURE;
  }

#ifdef WITH_CHROMA_ZMQ
  fPhotonList = new ChromaPhotonList ;   
  fZMQRoot = new ZMQRoot("CSA_CLIENT_CONFIG") ; 
#endif
  
  return StatusCode::SUCCESS; 
}

StatusCode DsChromaStackAction::finalize() 
{
  info() << "DsChromaStackAction::finalize()" << endreq;
  neutronList.clear();  

#ifdef WITH_CHROMA_ZMQ
  if(fPhotonList2) delete fPhotonList2 ; 
  delete fPhotonList ;
  delete fZMQRoot ;
#endif

  return  GiGaStackActionBase::finalize();
}


void DsChromaStackAction::CollectPhoton(const G4Track* aPhoton )
{
#ifdef WITH_CHROMA_ZMQ
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ;
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ;
   G4ThreeVector pol = aPhoton->GetPolarization() ;
   float time = aPhoton->GetGlobalTime()/ns ;
   float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   //fPhotonList->AddPhoton( pos, dir, pol, time, wavelength );
   // more appropriate for transport class to have a minimum of dependencies
   fPhotonList->AddPhoton( 
              pos.x(), pos.y(), pos.z(),
              dir.x(), dir.y(), dir.z(),
              pol.x(), pol.y(), pol.z(), 
              time, 
              wavelength );

#endif
}


//--------------------------------------------------------------------------

G4ClassificationOfNewTrack DsChromaStackAction::ClassifyNewTrack (const G4Track* aTrack) 
{
  G4ClassificationOfNewTrack classification = fUrgent;

  G4ParticleDefinition* definition = aTrack->GetDefinition() ;
  G4bool is_optical = definition == G4OpticalPhoton::OpticalPhotonDefinition() ;
  G4bool is_neutron = definition == G4Neutron::NeutronDefinition() ;
  G4bool is_gamma   = definition == G4Gamma::GammaDefinition() ;

  G4int trackid = aTrack->GetTrackID() ;
  G4int parentid = aTrack->GetParentID() ;
  G4bool is_direct  = trackid - parentid == 1 ;
  G4bool is_secondary = parentid > 0 ; 

  //
  // below "logic" (including repetitions) is a slavish copy of predecessor code
  // but expressed in a much more compact manner
  //
  // TODO: translate below highly break-able code into something more reasonable
  //
  
  switch(stage)
    {
    case 0: 

          if(!is_optical){

              if(m_tightCut){

                  if(is_neutron)
                  {
                      NeutronNumbers++;
                      neutronList.push_back(trackid); 
                      break ;
                  }

	              if( is_gamma && is_direct )
                  {
                      if(!interestingEvt){
                          interestingEvt=this->IsRelevantNeutronDaughter(aTrack);
                      }
                      break ;
                  }

	          } else {  // not tight 

                  if(is_neutron){
                      NeutronNumbers++;
                      break ;
                  }

                  if( !is_optical ) 
                  {
                      if(!interestingEvt){
                          interestingEvt=this->IsRelevant(aTrack);
                      }
                      break ;
	              }
              }


          } else {         // optical  
	
	          PhotonNumbers++;
              if(PhotonNumbers % m_moduloPhoton == 0){
                  CollectPhoton( aTrack );
              }

	          if (m_photonCut) 
              {
	              classification=fKill;
                  break;
	          }
	          else if( is_secondary && PhotonNumbers <= m_maxPhoton && !interestingEvt )  
              // keep em waiting, until interesting is flagged
	          {
                  classification=fWaiting;
                  break;
	          }
          }

          classification = fUrgent;
          break;
    case 1:
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

void DsChromaStackAction::NewStage()
{

  info()<< " StackingAction::NewStage! "<<endreq;
  info()<< " Number of Optical Photons generated:  "<< PhotonNumbers<<endreq;
  info()<< " Number of Neutron generated:  "<< NeutronNumbers<<endreq;


#ifdef WITH_CHROMA_ZMQ

  G4RunManager* runMan = G4RunManager::GetRunManager(); 
  const G4Event* currentEvent = runMan->GetCurrentEvent(); 
  G4int eventID = currentEvent->GetEventID();

  fPhotonList->SetUniqueID(eventID);
  info() << "::NewStage fPhotonList " <<  endreq ;   
  fPhotonList->Print(); 
  std::size_t size = fPhotonList->GetSize(); 

  if(size > 0)
  {
      info() << "::NewStage SendObject " <<  endreq ;   
      fZMQRoot->SendObject(fPhotonList);
      info() << "::NewStage ReceiveObject, waiting... " <<  endreq;   
      fPhotonList2 = (ChromaPhotonList*)fZMQRoot->ReceiveObject();
      info() << "::NewStage fPhotonList2 " <<  endreq ;   
      fPhotonList2->Print();
  } 
  else 
  { 
      info() << "::NewStage Skip send/recv for empty CPL " <<  endreq;   
  }

#endif


  
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
void DsChromaStackAction::PrepareNewEvent()
{
  info()<< " StackingAction::PrepareNewEvent "<<endreq;
  interestingEvt=false;
  stage=0;
  PhotonNumbers=0;
  NeutronNumbers=0;
  neutronList.clear();

  if(fPhotonList){ 
      info()<< " StackingAction::PrepareNewEvent fPhotonList ClearAll  "<<endreq;
      fPhotonList->ClearAll(); 
  }

  if(fPhotonList2){ 
      info()<< " StackingAction::PrepareNewEvent fPhotonList2 ClearAll  "<<endreq;
      fPhotonList2->ClearAll(); 
  } 

}

//-----------------If the Gamma neutron's daughter ? ---------------------

G4bool DsChromaStackAction::IsNeutronDaughter(const G4int id, const vector<G4int> aList)
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

G4bool DsChromaStackAction::IsRelevantNeutronDaughter(const G4Track* aTrack)
{
  G4int trkID=aTrack->GetParentID(); 
  return IsNeutronDaughter(trkID, neutronList) && IsRelevant(aTrack) ;
}


G4bool DsChromaStackAction::IsRelevant(const G4Track* aTrack)  // original PossibleInterestingTrack
{
  IDetectorElement *de;
  Gaudi::XYZPoint gp(aTrack->GetPosition().x(),aTrack->GetPosition().y(),aTrack->GetPosition().z());
  de = m_csvc->coordSysDE(gp);
  if(de){
      IGeometryInfo *ginfo=de->geometry();
      if(ginfo){
	      const ILVolume *lv=ginfo->lvolume();
	      if(lv){
	           G4String MaterialName = lv->materialName();
	  
	           if( MaterialName=="/dd/Materials/MineralOil"
	            || MaterialName== "/dd/Materials/GdDopedLS"
	            || MaterialName== "/dd/Materials/LiquidScintillator" 
	            || MaterialName== "/dd/Materials/Acrylic") {
	    
	               return true;
	            }
	      }
      }
  }
  return false;
}


