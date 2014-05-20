DetSim DsConsOptical
=======================

Geant4 Refs
-----------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/Overview/html/index.html

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/GettingStarted/physicsDef.html

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForToolkitDeveloper/html/index.html

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForToolkitDeveloper/html/node8.html



Geant4 Process
~~~~~~~~~~~~~~~~~

* http://www-geant4.kek.jp/g4users/tutorial03/material/4-1.pdf

* http://conferences.fnal.gov/g4tutorial/g4cd/Slides/Fermilab/PhysicsTutor2.pdf



Process Model
~~~~~~~~~~~~~~~


AtRest
   * AtRestDoIt, 
   * AtRestGetPhysicalInteractionLength
   * (G4VAtRestProcess, processes with only AtRestDoIt)

AlongStep
   * AlongStepDoIt, 
   * AlongStepGetPhysicalInteractionLength
   * (G4VContinuousProcess, processes with only AlongStepDoIt)

PostStep
   * PostStepDoIt
   * PostStepGetPhysicalInteractionLength
   * (G4VDiscreteProcess, processes with only PostStepDoIt)

Process Ordering
~~~~~~~~~~~~~~~~~

ProcessManager of a particle is responsible for providing the Ordering

#. Ordering of GetPhysicalInteractionLength in AlongStepDoIt,

   * Multiple Scattering process has to be invoked just before Transportation
   * Transportation process has to be invoked at the end.  

#. Ordering of DoIts 
 
   * There is only some special cases. 
   * For example, the Cherenkov process needs the energy loss information of
     the current step for its DoIt invocation. Therefore, the EnergyLoss process has
     to be invoked before the Cherenkov process. This ordering is provided by the
     process manager. Energy loss information necessary for the Cherenkov process is
     passed using G4Step (or the static dE/dX table is used together with the step
     length information in G4Step to obtain the energy loss information). Any other?


DsPhysConsOptical.h
---------------------

::

     //    Construct Optical Processes
     20 class DsPhysConsOptical : public GiGaPhysConstructorBase
     21 {

DsPhysConsOptical.cc
---------------------

Assuming `DsPhysConsOptical` is used `DsG4Scintillation` is being used by default define::

      01 #define USE_CUSTOM_CERENKOV
      02 #define USE_CUSTOM_SCINTILLATION
      03 
      04 #include "DsPhysConsOptical.h"
      05 #include "DsG4OpRayleigh.h"
      06 
      07 #ifdef USE_CUSTOM_CERENKOV
      08 #include "DsG4Cerenkov.h"
      09 #else
      10 #include "G4Cerenkov.hh"
      11 #endif
      12 
      13 #ifdef USE_CUSTOM_SCINTILLATION
      14 #include "DsG4Scintillation.h"
      15 #else
      16 #include "G4Scintillation.hh"
      17 #endif
      18 


Default properties::

     37     declareProperty("CerenMaxPhotonsPerStep",m_cerenMaxPhotonPerStep = 300,
     38                     "Limit step to at most this many (unscaled) Cerenkov photons.");
     39 
     40     declareProperty("ScintDoReemission",m_doReemission = true,
     41                     "Do reemission in scintilator.");
     42     declareProperty("ScintDoScintAndCeren",m_doScintAndCeren = true,
     43                     "Do both scintillation and Cerenkov in scintilator.");
     44 
     45     declareProperty("UseCerenkov", m_useCerenkov=true,
     46                     "Use the Cerenkov process?");
     47     declareProperty("ApplyWaterQe", m_applyWaterQe=true,
     48                     "Apply QE for water cerenkov process when OP is created?"
     49                     "If it is true the CerenPhotonScaleWeight will be disabled in water,"
     50                     "but it still works for AD and others");
     51                     // wz: Maybe we can set the weight of a OP to >1 in future.
     52 
     53     declareProperty("UseScintillation",m_useScintillation=true,
     54                     "Use the Scintillation process?");
     55     declareProperty("UseRayleigh", m_useRayleigh=true,
     56                     "Use the Rayleigh scattering process?");
     57     declareProperty("UseAbsorption", m_useAbsorption=true,
     58                     "Use light absorption process?");
     59     declareProperty("UseFastMu300nsTrick", m_useFastMu300nsTrick=false,
     60                     "Use Fast muon simulation?");
     61     declareProperty("ScintillationYieldFactor",m_ScintillationYieldFactor = 1.0,
     62             "Scale the number of scintillation photons per MeV by this much.");
     63 
     64     declareProperty("BirksConstant1", m_birksConstant1 = 6.5e-3*g/cm2/MeV,
     65                     "Birks constant C1");
     66     declareProperty("BirksConstant2", m_birksConstant2 = 3.0e-6*(g/cm2/MeV)*(g/cm2/MeV),
     67                    "Birks constant C2");
     68 
     69     declareProperty("GammaSlowerTime", m_gammaSlowerTime = 149*ns,
     70                     "Gamma Slower time constant");
     71     declareProperty("GammaSlowerRatio", m_gammaSlowerRatio = 0.338,
     72                    "Gamma Slower time ratio");
     73 
     74     declareProperty("NeutronSlowerTime", m_neutronSlowerTime = 220*ns,
     75                     "Neutron Slower time constant");
     76     declareProperty("NeutronSlowerRatio", m_neutronSlowerRatio = 0.34,
     77                    "Neutron Slower time ratio");
     78 
     79     declareProperty("AlphaSlowerTime", m_alphaSlowerTime = 220*ns,
     80                     "Alpha Slower time constant");
     81     declareProperty("AlphaSlowerRatio", m_alphaSlowerRatio = 0.35,
     82                    "Alpha Slower time ratio");
     83 
     84     declareProperty("CerenPhotonScaleWeight",m_cerenPhotonScaleWeight = 3.125,
     85                     "Scale down number of produced Cerenkov photons by this much.");
     86     declareProperty("ScintPhotonScaleWeight",m_scintPhotonScaleWeight = 3.125,
     87                     "Scale down number of produced scintillation photons by this much.");




::

    137 #ifdef USE_CUSTOM_SCINTILLATION
    138     DsG4Scintillation* scint = 0;
    139     info() << "Using customized DsG4Scintillation." << endreq;
    140     scint = new DsG4Scintillation();
    141     scint->SetBirksConstant1(m_birksConstant1);
    142     scint->SetBirksConstant2(m_birksConstant2);
    143     scint->SetGammaSlowerTimeConstant(m_gammaSlowerTime);
    144     scint->SetGammaSlowerRatio(m_gammaSlowerRatio);
    145     scint->SetNeutronSlowerTimeConstant(m_neutronSlowerTime);
    146     scint->SetNeutronSlowerRatio(m_neutronSlowerRatio);
    147     scint->SetAlphaSlowerTimeConstant(m_alphaSlowerTime);
    148     scint->SetAlphaSlowerRatio(m_alphaSlowerRatio);
    149     scint->SetDoReemission(m_doReemission);
    150     scint->SetDoBothProcess(m_doScintAndCeren);
    151     scint->SetApplyPreQE(m_scintPhotonScaleWeight>1.);
    152     scint->SetPreQE(1./m_scintPhotonScaleWeight);
    153     scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); //1.);
    154     scint->SetUseFastMu300nsTrick(m_useFastMu300nsTrick);
    155     scint->SetTrackSecondariesFirst(true);
    156     if (!m_useScintillation) {
    157         scint->SetNoOp();
    158     }
    159 #else  // standard G4 scint
    160     G4Scintillation* scint = 0;
    161     if (m_useScintillation) {
    162         info() << "Using standard G4Scintillation." << endreq;
    163         scint = new G4Scintillation();
    164         scint->SetScintillationYieldFactor(m_ScintillationYieldFactor); // 1.);
    165         scint->SetTrackSecondariesFirst(true);
    166     }
    167 #endif




`DsPhysConsOptical::ConstructProcess`
---------------------------------------

Creates and configures process instances:


#. cerenkov (DsG4Cerenkov)
#. scint (DsG4Scintillation)

#. absorb (G4OpAbsorption)
#. rayleigh (DsG4OpRayleigh)
#. boundproc (DsG4OpBoundaryProcess)
#. `fast_sim_man` (G4FastSimulationManagerProcess)

Adds the processes to particles to which they are applicable.
Last 4 are for Optical Photon and added with `AddDiscreteProcess`


Q: Is `G4Transportation` automatically added ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:src blyth$ grep Transportation *.*
    DsG4OpBoundaryProcess.cc:                     G4TransportationManager::GetTransportationManager()->
    DsG4OpBoundaryProcess.h:#include "G4TransportationManager.hh"
    DsPhysConsGeneral.cc:    // AddTransportation();
    DsPmtModel.cc:#include "G4TransportationManager.hh"
    DsRpcModel.cc:#include "G4TransportationManager.hh"



TransportationManager used to get Navigator in boundproc::

     134 G4VParticleChange*
     135 DsG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     136 {
     ...
     164         G4Navigator* theNavigator =
     165                      G4TransportationManager::GetTransportationManager()->
     166                                               GetNavigatorForTracking();
     167 
     168         G4ThreeVector theLocalPoint = theNavigator->
     169                                       GetGlobalToLocalTransform().
     170                                       TransformPoint(theGlobalPoint);


Comment indicates GiGa adds the Transportation::

    098 void DsPhysConsGeneral::ConstructProcess()
    099 {
    100     // can't call this from a GiGaPhysConstructorBase, but
    101     // G4VModularPhysicsList will do it for us.
    102     // AddTransportation();
    103 
    104     G4Decay* theDecayProcess = new G4Decay();
    105     theParticleIterator->reset();
    106     while( (*theParticleIterator)() ) {
    107         G4ParticleDefinition* particle = theParticleIterator->value();
    108         G4ProcessManager* pmanager = particle->GetProcessManager();
    109         if (theDecayProcess->IsApplicable(*particle)) {
    110             pmanager->AddDiscreteProcess(theDecayProcess);
    111             pmanager ->SetProcessOrdering(theDecayProcess, idxPostStep);
    112             pmanager ->SetProcessOrdering(theDecayProcess, idxAtRest);
    113         }
    114     }

Good to know where Optical Photon fake PDG code 20022 is set::

     43 void DsPhysConsGeneral::ConstructParticle()
     ..
     84     /// Special hook: change the PDG encoded value for optical photons to be unique and useful.
     85     G4ParticlePropertyTable* propTable = G4ParticlePropertyTable::GetParticlePropertyTable();
     86     assert(propTable);
     87     G4ParticlePropertyData* photonData = propTable->GetParticleProperty(G4OpticalPhoton::Definition());
     88     assert(photonData);
     89     photonData->SetPDGEncoding(20022);
     90     photonData->SetAntiPDGEncoding(20022);
     91     if(propTable->SetParticleProperty(*photonData))
     92       info() << "Set PDG code for opticalphoton to " << G4OpticalPhoton::Definition()->GetPDGEncoding() << endreq;
     93     else
     94       warning() << "Failed to reset PDG code on opticalphoton.. it's still set to "
     95                 << G4OpticalPhoton::Definition()->GetPDGEncoding() << endreq;


More Phys setup ?::

    delta:src blyth$ grep \ GiGaPhysConstructorBase *.h
    DsPhysConsEM.h:class DsPhysConsEM : public GiGaPhysConstructorBase
    DsPhysConsElectroNu.h:class DsPhysConsElectroNu : public GiGaPhysConstructorBase
    DsPhysConsGeneral.h:class DsPhysConsGeneral : public GiGaPhysConstructorBase
    DsPhysConsHadron.h:class DsPhysConsHadron : public GiGaPhysConstructorBase
    DsPhysConsIon.h:class DsPhysConsIon : public GiGaPhysConstructorBase
    DsPhysConsOptical.h:class DsPhysConsOptical : public GiGaPhysConstructorBase







`source/processes/parameterisation/include/G4FastSimulationManagerProcess.hh`
------------------------------------------------------------------------------

* `Fast` used to be `Parametrisation`
* http://geant4.slac.stanford.edu/UsersWorkshop/PDF/Marc/FastSimulation.pdf

G4FastSimulationManagerProcess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A `G4VProcess` (usaully Discrete) providing interface between tracking and fast simulation.

* It has to be set to the particles to be parameterised:

The process ordering:

* [n-3] ...
* [n-2] Multiple Scattering
* [n-1] G4FastSimulationManagerProcess 
* [ n ] G4Transportation


