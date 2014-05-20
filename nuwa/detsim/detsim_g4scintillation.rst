DetSim G4Scintillation
========================

Observations
--------------

#. not-applicable to optical photons

   * what about REEMISSION ?

#. g4ten includes Birks Saturation handling 


Basis properties:

#. FASTCOMPONENT
#. SLOWCOMPONENT
#. SCINTILLATIONYIELD   (scintillation photons/MeV)


Optical Photon production handles by setting secondaries::

    aParticleChange.SetNumberOfSecondaries(NumPhotons);

Then creating secondary `G4Track` with parameters based on boatload of properties and then `AddSecondary`::

     aParticleChange.AddSecondary(aSecondaryTrack);

Boatload:

#. FASTTIMECONSTANT
#. FASTSCINTILLATIONRISETIME
#. SLOWTIMECONSTANT
#. SLOWSCINTILLATIONRISETIME
#. YIELDRATIO



Latest geant4.10.00.p01 G4Scintillation
--------------------------------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html


`geant4.10.00.p01/source/processes/electromagnetic/xrays/include/G4Scintillation.hh`::

    088 class G4Scintillation : public G4VRestDiscreteProcess
    089 {
    ...
    192         static void AddSaturation(G4EmSaturation* );
    193         // Adds Birks Saturation to the process.
    194 
    195         static void RemoveSaturation();
    196         // Removes the Birks Saturation from the process.
    197 
    198         G4EmSaturation* GetSaturation() const { return fEmSaturation; }
    199         // Returns the Birks Saturation.
    ///
    ///   new addition ?
    ///
    255 inline
    256 G4bool G4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
    257 {
    258        if (aParticleType.GetParticleName() == "opticalphoton") return false;
    259        if (aParticleType.IsShortLived()) return false;
    260 
    261        return true;
    262 }


`geant4.10.00.p01/source/processes/electromagnetic/xrays/src/G4Scintillation.cc`::

    089 G4bool G4Scintillation::fTrackSecondariesFirst = false;
    090 G4bool G4Scintillation::fFiniteRiseTime = false;
    091 G4double G4Scintillation::fYieldFactor = 1.0;
    092 G4double G4Scintillation::fExcitationRatio = 1.0;
    093 G4bool G4Scintillation::fScintillationByParticleType = false;
    094 G4EmSaturation* G4Scintillation::fEmSaturation = NULL;
    ...
    ...
    169 G4VParticleChange*
    170 G4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    171 
    172 // This routine is called for each tracking step of a charged particle
    173 // in a scintillator. A Poisson/Gauss-distributed number of photons is
    174 // generated according to the scintillation yield formula, distributed
    175 // evenly along the track segment and uniformly into 4pi.
    176 
    177 {
    ...
    197         G4MaterialPropertyVector* Fast_Intensity =
    198                 aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    199         G4MaterialPropertyVector* Slow_Intensity =
    200                 aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    201 
    202         if (!Fast_Intensity && !Slow_Intensity )
    203              return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ///
    ///      requires a bunch of property tables to be defined 
    ///
    204 
    205         G4int nscnt = 1;
    206         if (Fast_Intensity && Slow_Intensity) nscnt = 2;
    207 
    208         G4double ScintillationYield = 0.;
    209 
    210     // Scintillation depends on particle type, energy deposited
    211         if (fScintillationByParticleType) {
    212 
    213            ScintillationYield =
    214                           GetScintillationYieldByParticleType(aTrack, aStep);
    215 
    216            // The default linear scintillation process
    217         } else {
    218 
    219            ScintillationYield = aMaterialPropertiesTable->
    220                                       GetConstProperty("SCINTILLATIONYIELD");
    221 
    222            // Units: [# scintillation photons / MeV]
    223            ScintillationYield *= fYieldFactor;
    224         }
    225 
    226         G4double ResolutionScale    = aMaterialPropertiesTable->
    227                                       GetConstProperty("RESOLUTIONSCALE");
    228 
    ///
    ///
    ///
    235         G4double MeanNumberOfPhotons;
    236 
    237         // Birk's correction via fEmSaturation and specifying scintillation by
    238         // by particle type are physically mutually exclusive
    239 
    240         if (fScintillationByParticleType)
    241            MeanNumberOfPhotons = ScintillationYield;
    242         else if (fEmSaturation)
    243            MeanNumberOfPhotons = ScintillationYield*
    244                               (fEmSaturation->VisibleEnergyDeposition(&aStep));
    245         else
    246            MeanNumberOfPhotons = ScintillationYield*TotalEnergyDeposit;
    247 
    248         G4int NumPhotons;
    249 
    250         if (MeanNumberOfPhotons > 10.)
    251         {
    252           G4double sigma = ResolutionScale * std::sqrt(MeanNumberOfPhotons);
    253           NumPhotons = G4int(G4RandGauss::shoot(MeanNumberOfPhotons,sigma)+0.5);
    254         }
    255         else
    256         {
    257           NumPhotons = G4int(G4Poisson(MeanNumberOfPhotons));
    258         }
    259 
    260         if (NumPhotons <= 0)
    261         {
    262            // return unchanged particle and no secondaries
    263 
    264            aParticleChange.SetNumberOfSecondaries(0);
    265 
    266            return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    267         }
    268 
    269         ////////////////////////////////////////////////////////////////
    270 
    271         aParticleChange.SetNumberOfSecondaries(NumPhotons);
    272 
    273         if (fTrackSecondariesFirst) {
    274            if (aTrack.GetTrackStatus() == fAlive )
    275                   aParticleChange.ProposeTrackStatus(fSuspend);
    276         }
    ...
    ...
    ...
    441                 G4double aSecondaryTime = t0 + deltaTime;
    442 
    443                 G4ThreeVector aSecondaryPosition =
    444                                     x0 + rand * aStep.GetDeltaPosition();
    445 
    446                 G4Track* aSecondaryTrack = new G4Track(aScintillationPhoton,
    447                                                        aSecondaryTime,
    448                                                        aSecondaryPosition);
    449 
    450                 aSecondaryTrack->SetTouchableHandle(
    451                                  aStep.GetPreStepPoint()->GetTouchableHandle());
    452                 // aSecondaryTrack->SetTouchableHandle((G4VTouchable*)0);
    453 
    454                 aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    455 
    456                 aParticleChange.AddSecondary(aSecondaryTrack);







