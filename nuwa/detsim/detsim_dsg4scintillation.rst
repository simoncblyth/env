DetSim DsG4Scintillation
==========================

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



::

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





