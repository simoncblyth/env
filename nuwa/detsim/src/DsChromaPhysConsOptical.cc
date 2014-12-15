#define USE_CUSTOM_CERENKOV
#define USE_CUSTOM_SCINTILLATION

#include "DsChromaPhysConsOptical.h"
#include "DsChromaG4OpRayleigh.h"

#ifdef USE_CUSTOM_CERENKOV
#include "DsChromaG4Cerenkov.h"
#else
#include "G4Cerenkov.hh"
#endif

#ifdef USE_CUSTOM_SCINTILLATION
#include "DsChromaG4Scintillation.h"
#else
#include "G4Scintillation.hh"
#endif

#include "GaudiKernel/DeclareFactoryEntries.h" 
#include "GaudiKernel/IProperty.h" 

#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
//#include "G4OpBoundaryProcess.hh"
#include "DsChromaG4OpBoundaryProcess.h"
#include "G4ProcessManager.hh"
#include "G4FastSimulationManagerProcess.hh"

DECLARE_TOOL_FACTORY(DsChromaPhysConsOptical);


DsChromaPhysConsOptical::DsChromaPhysConsOptical(const std::string& type,
                                     const std::string& name, 
                                     const IInterface* parent)
    : GiGaPhysConstructorBase(type,name,parent)
{
    declareProperty("CerenMaxPhotonsPerStep",m_cerenMaxPhotonPerStep = 300,
                    "Limit step to at most this many (unscaled) Cerenkov photons.");

    declareProperty("ScintDoReemission",m_doReemission = true,
                    "Do reemission in scintilator.");
    declareProperty("ScintDoScintAndCeren",m_doScintAndCeren = true,
                    "Do both scintillation and Cerenkov in scintilator.");

    declareProperty("UseCerenkov", m_useCerenkov=true, 
                    "Use the Cerenkov process?");
    declareProperty("ApplyWaterQe", m_applyWaterQe=true,
                    "Apply QE for water cerenkov process when OP is created?"
                    "If it is true the CerenPhotonScaleWeight will be disabled in water,"
                    "but it still works for AD and others");  
                    // wz: Maybe we can set the weight of a OP to >1 in future.

    declareProperty("UseScintillation",m_useScintillation=true,
                    "Use the Scintillation process?");
    declareProperty("UseRayleigh", m_useRayleigh=true,
                    "Use the Rayleigh scattering process?");
    declareProperty("UseAbsorption", m_useAbsorption=true,
                    "Use light absorption process?");
    declareProperty("UseFastMu300nsTrick", m_useFastMu300nsTrick=false,
                    "Use Fast muon simulation?");
    declareProperty("ScintillationYieldFactor",m_ScintillationYieldFactor = 1.0,
		    "Scale the number of scintillation photons per MeV by this much.");
   
    declareProperty("BirksConstant1", m_birksConstant1 = 6.5e-3*g/cm2/MeV, 
                    "Birks constant C1");
    declareProperty("BirksConstant2", m_birksConstant2 = 3.0e-6*(g/cm2/MeV)*(g/cm2/MeV), 
                   "Birks constant C2");

    declareProperty("GammaSlowerTime", m_gammaSlowerTime = 149*ns, 
                    "Gamma Slower time constant");
    declareProperty("GammaSlowerRatio", m_gammaSlowerRatio = 0.338, 
                   "Gamma Slower time ratio");

    declareProperty("NeutronSlowerTime", m_neutronSlowerTime = 220*ns, 
                    "Neutron Slower time constant");
    declareProperty("NeutronSlowerRatio", m_neutronSlowerRatio = 0.34, 
                   "Neutron Slower time ratio");

    declareProperty("AlphaSlowerTime", m_alphaSlowerTime = 220*ns, 
                    "Alpha Slower time constant");
    declareProperty("AlphaSlowerRatio", m_alphaSlowerRatio = 0.35, 
                   "Alpha Slower time ratio");
     
    declareProperty("CerenPhotonScaleWeight",m_cerenPhotonScaleWeight = 3.125,
                    "Scale down number of produced Cerenkov photons by this much."); 
    declareProperty("ScintPhotonScaleWeight",m_scintPhotonScaleWeight = 3.125, 	 	 
                    "Scale down number of produced scintillation photons by this much."); 
}

StatusCode DsChromaPhysConsOptical::initialize()
{
    info()<<"Photons prescaling is "<<( m_cerenPhotonScaleWeight>1.?"on":"off" )
          <<" for Cerenkov. Preliminary applied efficiency is "
          <<1./m_cerenPhotonScaleWeight<<" (weight="<<m_cerenPhotonScaleWeight<<")"<<endreq;
    info()<<"Photons prescaling is "<<( m_scintPhotonScaleWeight>1.?"on":"off" )
          <<" for Scintillation. Preliminary applied efficiency is "
          <<1./m_scintPhotonScaleWeight<<" (weight="<<m_scintPhotonScaleWeight<<")"<<endreq;
    info()<<"WaterQE is turned "<<(m_applyWaterQe?"on":"off")<<" for Cerenkov."<<endreq;
    return this->GiGaPhysConstructorBase::initialize();
}

DsChromaPhysConsOptical::~DsChromaPhysConsOptical()
{
}

void DsChromaPhysConsOptical::ConstructParticle()
{
}

void DsChromaPhysConsOptical::ConstructProcess()
{
#ifdef USE_CUSTOM_CERENKOV
    
    info () << "Using customized DsChromaG4Cerenkov." << endreq;
    DsChromaG4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) {
        cerenkov = new DsChromaG4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetApplyPreQE(m_cerenPhotonScaleWeight>1.);
        cerenkov->SetPreQE(1./m_cerenPhotonScaleWeight);
        
        // wangzhe   Give user a handle to control it.   
        cerenkov->SetApplyWaterQe(m_applyWaterQe);
        // wz
        cerenkov->SetTrackSecondariesFirst(true);
    }
#else
    info () << "Using standard G4Cerenkov." << endreq;
    G4Cerenkov* cerenkov = 0;
    if (m_useCerenkov) {
        cerenkov = new G4Cerenkov();
        cerenkov->SetMaxNumPhotonsPerStep(m_cerenMaxPhotonPerStep);
        cerenkov->SetTrackSecondariesFirst(true);
    }
#endif

#ifdef USE_CUSTOM_SCINTILLATION
    DsChromaG4Scintillation* scint = 0;
    info() << "Using customized DsChromaG4Scintillation." << endreq;
    scint = new DsChromaG4Scintillation();
    scint->SetBirksConstant1(m_birksConstant1);
    scint->SetBirksConstant2(m_birksConstant2);
    scint->SetGammaSlowerTimeConstant(m_gammaSlowerTime);
    scint->SetGammaSlowerRatio(m_gammaSlowerRatio);
    scint->SetNeutronSlowerTimeConstant(m_neutronSlowerTime);
    scint->SetNeutronSlowerRatio(m_neutronSlowerRatio);
    scint->SetAlphaSlowerTimeConstant(m_alphaSlowerTime);
    scint->SetAlphaSlowerRatio(m_alphaSlowerRatio);
    scint->SetDoReemission(m_doReemission);
    scint->SetDoBothProcess(m_doScintAndCeren);
    scint->SetApplyPreQE(m_scintPhotonScaleWeight>1.);
    scint->SetPreQE(1./m_scintPhotonScaleWeight);
    scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); //1.);
    scint->SetUseFastMu300nsTrick(m_useFastMu300nsTrick);
    scint->SetTrackSecondariesFirst(true);
    if (!m_useScintillation) {
        scint->SetNoOp();
    }
#else  // standard G4 scint
    G4Scintillation* scint = 0;
    if (m_useScintillation) {
        info() << "Using standard G4Scintillation." << endreq;
        scint = new G4Scintillation();
        scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); // 1.);
        scint->SetTrackSecondariesFirst(true);
    }
#endif

    G4OpAbsorption* absorb = 0;
    if (m_useAbsorption) {
        absorb = new G4OpAbsorption();
    }

    DsChromaG4OpRayleigh* rayleigh = 0;
    if (m_useRayleigh) {
        rayleigh = new DsChromaG4OpRayleigh();
	//        rayleigh->SetVerboseLevel(2);
    }

    //G4OpBoundaryProcess* boundproc = new G4OpBoundaryProcess();
    DsChromaG4OpBoundaryProcess* boundproc = new DsChromaG4OpBoundaryProcess();
    boundproc->SetModel(unified);

    G4FastSimulationManagerProcess* fast_sim_man
        = new G4FastSimulationManagerProcess("fast_sim_man");
    
    theParticleIterator->reset();
    while( (*theParticleIterator)() ) {

        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
    
        // Caution: as of G4.9, Cerenkov becomes a Discrete Process.
        // This code assumes a version of G4Cerenkov from before this version.

        if(cerenkov && cerenkov->IsApplicable(*particle)) {
            pmanager->AddProcess(cerenkov);
            pmanager->SetProcessOrdering(cerenkov, idxPostStep);
            debug() << "Process: adding Cherenkov to " 
                    << particle->GetParticleName() << endreq;
        }

        if(scint && scint->IsApplicable(*particle)) {
            pmanager->AddProcess(scint);
            pmanager->SetProcessOrderingToLast(scint, idxAtRest);
            pmanager->SetProcessOrderingToLast(scint, idxPostStep);
            debug() << "Process: adding Scintillation to "
                    << particle->GetParticleName() << endreq;
        }

        if (particle == G4OpticalPhoton::Definition()) {
            if (absorb)
                pmanager->AddDiscreteProcess(absorb);
            if (rayleigh)
                pmanager->AddDiscreteProcess(rayleigh);
            pmanager->AddDiscreteProcess(boundproc);
            //pmanager->AddDiscreteProcess(pee);
            pmanager->AddDiscreteProcess(fast_sim_man);
        }
    }
}
