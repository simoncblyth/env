/**
 * \class DsPhysConsOptical
 *
 * \brief Construct Optical Processes
 *
 * 
 *
 *
 *
 * bv@bnl.gov Tue Jan 29 15:25:22 2008
 *
 */


#ifndef DSPHYSCONSOPTICAL_H
#define DSPHYSCONSOPTICAL_H

#include "GiGa/GiGaPhysConstructorBase.h"

class DsPhysConsOptical : public GiGaPhysConstructorBase
{

public:

    DsPhysConsOptical(const std::string& type,
                     const std::string& name,
                     const IInterface* parent);
    virtual ~DsPhysConsOptical();

    // Interface methods
    void ConstructParticle();
    void ConstructProcess();
    StatusCode initialize (); 
private:
    /// Property: UseCerenkov, UseScintillation, UseRayleigh, UseAbsorption
    /// Turn on/off optical processes.  Default, all are on.
    bool m_useCerenkov, m_useScintillation, m_useRayleigh, m_useAbsorption;

    // wangzhe: Apply QE for water cerenkov process when OP is created?
    // See DsG4Cerenkov and Doc 3925 for details
    bool m_applyWaterQe;

    /// Property: CerenPhotonScaleWeight
    /// Number (>= 1.0) used to scale down the mean number of optical
    /// photons produced.  For each actual secondary optical photon
    /// track produced, it will be given a weight equal to this scale
    /// for scaling up if detected later.  Default is 1.0.
    double m_cerenPhotonScaleWeight;

    /// Property: CerenMaxPhotonsPerStep
    /// Maximum number of photons per step to limit step size.  This
    /// value is independent from PhotonScaleWeight.  Default is 300.
    int m_cerenMaxPhotonPerStep;

    /// ScintPhotonScaleWeight: Scale down number of produced
    /// scintillation photons by this much
    double m_scintPhotonScaleWeight;


    /// ScintillationYieldFactor: scale the number of produced scintillation
    /// photons per MeV by this much.
    /// This controls the primary yield of scintillation photons per MeV of
    /// deposited energy.
    double m_ScintillationYieldFactor;
    bool   m_useFastMu300nsTrick; 

    /// Birks constants C1 and C2
    double m_birksConstant1;
    double m_birksConstant2;

    double m_gammaSlowerTime;
    double m_gammaSlowerRatio;
    double m_neutronSlowerTime;
    double m_neutronSlowerRatio;
    double m_alphaSlowerTime;
    double m_alphaSlowerRatio;


    /// ScintDoReemission: Do reemission in scintilator
    bool m_doReemission;

    /// ScintDoScintAndCeren: Do both scintillation and Cerenkov in scintilator
    bool m_doScintAndCeren;
};

#endif  // DSPHYSCONSOPTICAL_H
