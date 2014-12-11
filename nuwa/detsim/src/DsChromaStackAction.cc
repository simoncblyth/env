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

#include "G4DAEChroma/G4DAEChroma.hh"


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
    m_csvc(0),
    m_chroma(0)
{ 
    declareProperty("NeutronParent",m_neutronParent = false, "Select events with neutron parent in addition to being in MO/LS/GdLS/Acrylic");
    declareProperty("PhotonKill",m_photonKill = false, " Kill all the optical photons in the process.");
    declareProperty("MaxPhoton",m_maxPhoton = 1e6, " Max number of photons to be hold.");
    declareProperty("ModuloPhoton",m_moduloPhoton = 100, "Modulo scale down photons collected.");
    declareProperty("ChromaPropagate",m_chromaPropagate = false, "Propagate optical photons externally, requires DsChromaRunAction instanciating Chroma.");
}

void DsChromaStackAction::Dump(const char* msg )
{
    info()<< "DsChromaStackAction::Dump " << msg <<endreq;
    info()<< " ========================================= " << endreq ; 
    info()<< " NeutronParent   :  "<< m_neutronParent  <<endreq;
    info()<< " PhotonKill      :  "<< m_photonKill  <<endreq;
    info()<< " MaxPhoton       :  "<< m_maxPhoton  <<endreq;
    info()<< " ModuloPhoton    :  "<< m_moduloPhoton  <<endreq;
    info()<< " ChromaPropagate :  "<< m_maxPhoton  <<endreq;
    info()<< " ========================================= " << endreq ; 
    info()<< " photonNumbers   :  "<< m_photonNumbers <<endreq;
    info()<< " neutronNumbers  :  "<< m_neutronNumbers <<endreq;
    info()<< " ========================================= " << endreq ; 
} 

StatusCode DsChromaStackAction::initialize() 
{
  info() << "DsChromaStackAction::initialize " << endreq;
  
  StatusCode sc = GiGaStackActionBase::initialize();
  if (sc.isFailure()) return sc;

  if ( service("CoordSysSvc", m_csvc).isFailure()) {
    error() << " No CoordSysSvc available." << endreq;
    return StatusCode::FAILURE;
  }
    
  if(m_chromaPropagate)
  {
#ifdef WITH_CHROMA_ZMQ
      info() << "DsChromaStackAction::initialize with Chroma propagation " << endreq;
      m_chroma = G4DAEChroma::GetG4DAEChroma();  // should have already been configured in RunAction TODO:check this
      m_chroma->Note("DsChromaStackAction::initialize"); 
      m_chroma->Print();
#else
      warn() << "DsChromaStackAction::initialize ChromaPropagate requested but not compiled -DWITH_CHROMA_ZMQ " << endreq;
#endif
  }
  else
  {
      info() << "DsChromaStackAction::initialize with standard Geant4 propagation" << endreq;
  }

  return StatusCode::SUCCESS; 
}

StatusCode DsChromaStackAction::finalize() 
{
  info() << "DsChromaStackAction::finalize()" << endreq;
  m_neutronList.clear();  

  if(m_chroma)
  {
      m_chroma->Note("DsChromaStackAction::finalize"); 
  } 

  return  GiGaStackActionBase::finalize();
}


//--------------------------------------------------------------------------

G4ClassificationOfNewTrack DsChromaStackAction::ClassifyNewTrack (const G4Track* aTrack) 
{
  G4ClassificationOfNewTrack classification = fUrgent;
  switch(m_stage)
  {
    case 0: 
          {
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
        
                  m_photonNumbers++;

                  // either kill all photons or modulo misses for non-realistic (but faster) testing 
                  if (m_photonKill || m_photonNumbers % m_moduloPhoton != 0) 
                  {
                      classification=fKill;
                  }
                  else if( is_secondary && m_photonNumbers <= m_maxPhoton && !m_interestingEvt )  
                  {
                      // Chroma:Collect+fKill === Geant4:fWaiting  
                      if( m_chroma ) 
                      { 
                          m_chroma->CollectPhoton( aTrack ); 
                          classification=fKill ;
                      } 
                      else
                      {
                          classification=fWaiting;    
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
  return classification;
}

//-------------------------------------------------------------------------
//-------------throw away non-interesting events, and-----------------------
//------------propogate optical photons otherwise ------------------
//-------------------------------------------------------------------

void DsChromaStackAction::NewStage()
{
  m_stage++;

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

      if(m_chroma)
      {
          m_chroma->Propagate(1); 
      }
      else
      {
          stackManager->ReClassify();  // stage>0 immediately goes to fUrgent
      }
  }
  else 
  {
      info()<< "Boring event, aborting..."<<endreq;
      if(m_chroma)
      {
          m_chroma->ClearAll();
      }  
      else
      {
          stackManager->clear();  //abort the event
      }
  }

}



//----------------------Reset -----------------------------------
void DsChromaStackAction::PrepareNewEvent()
{
  info()<< " StackingAction::PrepareNewEvent "<<endreq;
  m_interestingEvt=false;
  m_stage=0;
  m_photonNumbers=0;
  m_neutronNumbers=0;
  m_neutronList.clear();

  if(m_chroma)
  { 
      m_chroma->ClearAll(); 
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


