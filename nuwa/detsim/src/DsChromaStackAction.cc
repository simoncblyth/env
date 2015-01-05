#include "DsChromaStackAction.h"


#include "Conventions/Detectors.h"
#include "Event/SimPmtHit.h"

#include "DetDesc/IPVolume.h"
#include "DetHelpers/ICoordSysSvc.h"
#include "DetDesc/DetectorElement.h"
#include "DetDesc/IGeometryInfo.h"
#include "GaudiKernel/Service.h"
#include "GaudiKernel/DeclareFactoryEntries.h"

#include <G4ClassificationOfNewTrack.hh>
#include <G4RunManager.hh>
#include <G4TrackStatus.hh>
#include <G4ParticleDefinition.hh>
#include <G4Event.hh>
#include <G4ParticleTypes.hh>
#include <G4Track.hh>


#ifdef G4DAECHROMA
#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#endif


DECLARE_TOOL_FACTORY(DsChromaStackAction);

// duplicate parts of NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc 
// as parasite on the StackAction 
// as dont know how to access DsPmtSensDet private methods from StackAction 
//


DsChromaStackAction::DsChromaStackAction ( const std::string& type   , 
				   const std::string& name   , 
				   const IInterface*  parent ) 
  : GiGaStackActionBase( type, name, parent ) ,
    m_stage(0),
    m_photonNumbers(0),
    m_neutronNumbers(0),
    m_interestingEvt(false),
    m_csvc(0)
{ 
    declareProperty("NeutronParent",m_neutronParent = false, "Select events with neutron parent in addition to being in MO/LS/GdLS/Acrylic");
    declareProperty("PhotonKill",m_photonKill = false, " Kill all the optical photons in the process.");
    declareProperty("MaxPhoton",m_maxPhoton = 1e6, " Max number of photons to be hold.");
    declareProperty("ModuloPhoton",m_moduloPhoton = 100, "Modulo scale down photons collected.");
}

void DsChromaStackAction::Dump(const char* msg )
{
    info()<< "DsChromaStackAction::Dump " << msg <<endreq;
    info()<< " ========================================= " << endreq ; 
    info()<< " NeutronParent   :  "<< m_neutronParent  <<endreq;
    info()<< " PhotonKill      :  "<< m_photonKill  <<endreq;
    info()<< " MaxPhoton       :  "<< m_maxPhoton  <<endreq;
    info()<< " ModuloPhoton    :  "<< m_moduloPhoton  <<endreq;
    info()<< " ========================================= " << endreq ; 
    info()<< " photonNumbers   :  "<< m_photonNumbers <<endreq;
    info()<< " neutronNumbers  :  "<< m_neutronNumbers <<endreq;
    info()<< " ========================================= " << endreq ; 
} 

StatusCode DsChromaStackAction::initialize() 
{
  StatusCode sc = GiGaStackActionBase::initialize();
  if (sc.isFailure()) return sc;

  if ( service("CoordSysSvc", m_csvc).isFailure()) {
    error() << " No CoordSysSvc available." << endreq;
    return StatusCode::FAILURE;
  }
    
  return StatusCode::SUCCESS; 
}

StatusCode DsChromaStackAction::finalize() 
{
  m_neutronList.clear();  

  return  GiGaStackActionBase::finalize();
}


void DsChromaStackAction::PrepareNewEvent()
{
  info()<< " StackingAction::PrepareNewEvent "<<endreq;
  m_interestingEvt=false;
  m_stage=0;
  m_photonNumbers=0;
  m_neutronNumbers=0;
  m_neutronList.clear();

#ifdef G4DAECHROMA
  m_map.clear(); 
  G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
  chroma->Start("STACK"); 
  chroma->Stamp("PrepareNewEvent"); 
#endif
}



G4ClassificationOfNewTrack DsChromaStackAction::ClassifyNewTrack (const G4Track* aTrack) 
{

#ifdef G4DAECHROMA
  G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
  size_t STACK_CLASSIFY = chroma->FindFlag("STACK_CLASSIFY");
  size_t STACK_OP       = chroma->FindFlag("STACK_OP");
  size_t STACK_KILL     = chroma->FindFlag("STACK_KILL");
#endif

  G4ClassificationOfNewTrack classification = fUrgent;
  switch(m_stage)
  {
    case 0: 
          {
#ifdef G4DAECHROMA
              chroma->Register(STACK_CLASSIFY, 10000);
#endif

              G4ParticleDefinition* definition = aTrack->GetDefinition() ;
              G4bool is_optical = definition == G4OpticalPhoton::OpticalPhotonDefinition() ;
              G4bool is_neutron = definition == G4Neutron::NeutronDefinition() ;
              G4bool is_gamma   = definition == G4Gamma::GammaDefinition() ;

              G4int trackid = aTrack->GetTrackID() ;
              G4int parentid = aTrack->GetParentID() ;
              G4bool is_direct  = trackid - parentid == 1 ;
              G4bool is_secondary = parentid > 0 ; 


              if(!is_optical)
              {

                  if(m_neutronParent){

                      if(is_neutron)
                      {
                          m_neutronNumbers++;
                          m_neutronList.push_back(trackid); 
                          break ;
                      }

                      if( is_gamma && is_direct )
                      {
                          // only check for interestingness until its flagged 
                          // just one track of interest is enough for event to be regarded as interesting
                          if(!m_interestingEvt)
                          {
                              m_interestingEvt=this->IsRelevantNeutronDaughter(aTrack);
                          }
                          break ;
                      }

                  } else {  // not requiring neutronParent

                      if(is_neutron)
                      {
                          m_neutronNumbers++;
                          break ;
                      }

                      if( !is_optical ) 
                      {
                          if(!m_interestingEvt)
                          {
                              m_interestingEvt=this->IsRelevant(aTrack);
                          }
                          break ;
                      }
                  }


              } else {         // optical  
       
                  assert(is_secondary); 

#ifdef G4DAECHROMA
                  chroma->Register(STACK_OP);
#endif
                  m_photonNumbers++;

                  // either kill all photons or modulo misses for non-realistic (but faster) testing  or overmax
                  bool kill = m_photonKill || m_photonNumbers % m_moduloPhoton != 0 || m_photonNumbers >= m_maxPhoton ;

                  //
                  // FORMERLY: 
                  //       chroma always kills from G4 point of view, but collects non-kills
                  //
                  // BUT NOW:
                  //        chroma kills photons before they become G4Track within the Cerenkov and Scintillation
                  //        processes by not adding secondary,
                  //        instead process steps are collected and generated into photons on the GPU 
                  //
                  //

                  /*
                  if( m_chroma )
                  {
                       classification=fKill;  
                  }
                  else
                  */

                  { 
                       // once strike interestingness everything gets classified urgent
                       if(kill)
                       {
                           classification=fKill;
                       } 
                       else if( is_secondary && !m_interestingEvt )  
                       {
                           classification=fWaiting;    
                       }
                       else
                       {
                           //G4cout <<  *aTrack << G4endl ;
                       }
                  }
                  break;
              }

              classification = fUrgent;
          }
          break;
    case 1:
          classification = fUrgent;
          break;
    default:
          classification = fUrgent;
  } 


#ifdef G4DAECHROMA
  if(classification == fKill)
  {
      chroma->Register(STACK_KILL);
  }   
#endif

  return classification;
}

//-------------------------------------------------------------------------
//-------------throw away non-interesting events, and-----------------------
//------------propogate optical photons otherwise ------------------
//-------------------------------------------------------------------

void DsChromaStackAction::NewStage()
{
  m_stage++;

#ifdef G4DAECHROMA
  G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 
  chroma->Stop("STACK"); 
  chroma->Stamp("NewStage"); 

  m_map["photonNumbers"]  = toStr<int>(m_photonNumbers) ;  
  m_map["neutronNumbers"]  = toStr<int>(m_neutronNumbers) ;  
  m_map["moduloPhoton"]  = toStr<int>(m_moduloPhoton) ;  
  m_map["maxPhoton"]  = toStr<int>(m_maxPhoton) ;  

  m_map["COLUMNS"] = "NewStage:s,PrepareNewEvent:s,photonNumbers:i,neutronNumbers:i,moduloPhoton:i,maxPhoton:i" ;

  G4DAEMetadata* results = chroma->GetResults(); 
  results->AddMap("stackaction",m_map);

#endif

  info()<< "DsChromaStackAction::NewStage m_stage  " << m_stage <<endreq;
  info()<< " photonNumbers  :  "<< m_photonNumbers <<endreq;
  info()<< " neutronNumbers :  "<< m_neutronNumbers <<endreq;
  info()<< " moduloPhoton   :  "<< m_moduloPhoton  <<endreq;
  info()<< " maxPhoton      :  "<< m_maxPhoton  <<endreq;

  
  if(m_neutronParent)
  {
      info() << "neutronParent selection: only proceed with AD gamma events from neutrons " <<endreq;
  }
  else
  {
      info() << "proceed with events with at least on track in  MO/LS/GdLS/Acrylic" <<endreq;
  }


  if(m_photonNumbers>m_maxPhoton) 
  {
      info() << " m_photonNumbers > m_maxPhoton " << m_photonNumbers << " " << m_maxPhoton  <<endreq;
  }
  
  if(m_photonKill)
  {
       info() << "photonKill  All the Optical Photons killed in this event! " <<endreq;
  }

  
  if(m_interestingEvt)
  {
      info()<<" An interesting event! Let's go on!"<<endreq;

      /*
      if(m_chroma)
      {
          m_chroma->Propagate(1); 
      }
      else
      */
      {
          stackManager->ReClassify();  // stage>0 immediately goes to fUrgent
      }
  }
  else 
  {
      info()<< "Boring event, aborting..."<<endreq;

      /*
      if(m_chroma)
      {
          m_chroma->ClearAll();
      }  
      else
      */

      {
          stackManager->clear();  //abort the event
      }
  }

}



//-----------------If the Gamma neutron's daughter ? ---------------------

G4bool DsChromaStackAction::IsNeutronDaughter(const G4int id, const std::vector<G4int>& aList)
{
  //check if the gamma is the daughter of neutrons.
  G4bool isDaughter(false);
  for(size_t ii=0;ii<aList.size();ii++)
  {
      if(aList[ii]==id || aList[ii]==id-1) 
      {
          info()<<" neutron TrackID: "<<aList[ii]<< " been Captured! "<< endreq;
          isDaughter=true;
          break;
      }
  }
  return isDaughter;
}

// ------------- If the neutron been captured in the AD ? -------------------

G4bool DsChromaStackAction::IsRelevantNeutronDaughter(const G4Track* aTrack)
{
  G4int trkID=aTrack->GetParentID(); 
  return IsNeutronDaughter(trkID, m_neutronList) && IsRelevant(aTrack) ;
}


G4bool DsChromaStackAction::IsRelevant(const G4Track* aTrack)  // original PossibleInterestingTrack
{
  IDetectorElement *de;
  Gaudi::XYZPoint gp(aTrack->GetPosition().x(),aTrack->GetPosition().y(),aTrack->GetPosition().z());
  de = m_csvc->coordSysDE(gp);
  if(de)
  {
      IGeometryInfo *ginfo=de->geometry();
      if(ginfo)
      {
	      const ILVolume *lv=ginfo->lvolume();
	      if(lv)
          {
	          G4String MaterialName = lv->materialName();
	          if( MaterialName=="/dd/Materials/MineralOil"
	           || MaterialName=="/dd/Materials/GdDopedLS"
	           || MaterialName=="/dd/Materials/LiquidScintillator" 
	           || MaterialName=="/dd/Materials/Acrylic") 
              {
	              return true;
	          }
	      }
      }
  }
  return false;
}


