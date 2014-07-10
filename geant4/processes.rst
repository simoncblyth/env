Geant4 Processes
=================

Modelling Processes
--------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html

Each process has two groups of methods which play an important role in tracking, 

`GetPhysicalInteractionLength (GPIL)` 
        Get step length from the current space-time point to the next space-time point.
        It does this by calculating the probability of interaction based on the
        process's cross section information. At the end of this step the DoIt method
        should be invoked. 

`DoIt`. 
        The DoIt method implements the details of the interaction,
        changing the particle's energy, momentum, direction and position, and producing
        secondary tracks if required. These changes are recorded as G4VParticleChange
        objects(see Particle Change).



`G4VRestProcess`  
            processes using only the `AtRestDoIt` method, example: neutron capture
`G4VContinuousProcess`    
            processes using only the `AlongStepDoIt` method, example: cerenkov
`G4VDiscreteProcess`  
            processes using only the `PostStepDoIt` method, example: compton scattering, hadron inelastic interaction


OR for more complex processes which implement 2 or 3 of those 3 methods:
`G4VContinuousDiscreteProcess`, `G4VRestDiscreteProcess`, `G4VRestContinuousProcess`, `G4VRestContinuousDiscreteProcess`





Tracking of Photons in processes/optical
------------------------------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html

Absorption
~~~~~~~~~~~~

The implementation of optical photon bulk absorption, `G4OpAbsorption`, is
trivial in that the process merely kills the particle. The procedure requires
the user to fill the relevant `G4MaterialPropertiesTable` with empirical data for
the absorption length, using `ABSLENGTH` as the property key in the public method
AddProperty. The absorption length is the average distance traveled by a photon
before being absorpted by the medium; i.e. it is the mean free path returned by
the `GetMeanFreePath` method.

ABSLENGTH
^^^^^^^^^^

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H 'ABSLENGTH' {} \; 
    ./processes/optical/src/G4OpWLS.cc:      GetProperty("WLSABSLENGTH");
    ./processes/optical/src/G4OpAbsorption.cc:                                                GetProperty("ABSLENGTH");


G4OpAbsorption::PostStepDoIt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    101 G4VParticleChange*
    102 G4OpAbsorption::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    103 {
    104         aParticleChange.Initialize(aTrack);
    105 
    106         aParticleChange.ProposeTrackStatus(fStopAndKill);
    107 
    108         if (verboseLevel>0) {
    109        G4cout << "\n** Photon absorbed! **" << G4endl;
    110         }
    111         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    112 }


G4OpAbsorption::GetMeanFreePath
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    118 G4double G4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,
    119                          G4double ,
    120                          G4ForceCondition* )
    121 {
    122     const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    123         const G4Material* aMaterial = aTrack.GetMaterial();
    124 
    125     G4double thePhotonMomentum = aParticle->GetTotalMomentum();
    126 
    127     G4MaterialPropertiesTable* aMaterialPropertyTable;
    128     G4MaterialPropertyVector* AttenuationLengthVector;
    129 
    130         G4double AttenuationLength = DBL_MAX;
    131 
    132     aMaterialPropertyTable = aMaterial->GetMaterialPropertiesTable();
    133 
    134     if ( aMaterialPropertyTable ) {
    135        AttenuationLengthVector = aMaterialPropertyTable->
    136                                                 GetProperty("ABSLENGTH");
    137            if ( AttenuationLengthVector ){
    138              AttenuationLength = AttenuationLengthVector->
    139                                          GetProperty (thePhotonMomentum);




G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    (gdb) b 'G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)' 
    Breakpoint 1 at 0x68f34ed: file /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VDiscreteProcess.hh, line 137.

    (gdb) bt
    #0  G4VDiscreteProcess::PostStepGetPhysicalInteractionLength (this=0xd37b178, track=@0x10c8cd90, previousStepSize=0, condition=0xc481da0) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VDiscreteProcess.hh:137
    #1  0x07247e95 in G4VProcess::PostStepGPIL (this=0xd37b178, track=@0x10c8cd90, previousStepSize=0, condition=0xc481da0) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:464
    #2  0x0724655a in G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:165
    #3  0x07242e2c in G4SteppingManager::Stepping (this=0xc481c98) at src/G4SteppingManager.cc:181
    #4  0x0725150a in G4TrackingManager::ProcessOneTrack (this=0xc481c70, apValueG4Track=0x10c8cd90) at src/G4TrackingManager.cc:126
    #5  0xb666c24f in G4EventManager::DoProcessing (this=0xc481480, anEvent=0x10832350) at src/G4EventManager.cc:185
    #6  0xb666c9e6 in G4EventManager::ProcessOneEvent (this=0xc481480, anEvent=0x10832350) at src/G4EventManager.cc:335
    #7  0xb4e605e8 in GiGaRunManager::processTheEvent (this=0xc480c18) at ../src/component/GiGaRunManager.cpp:207
    #8  0xb4e5f522 in GiGaRunManager::retrieveTheEvent (this=0xc480c18, event=@0xbfa6c9d8) at ../src/component/GiGaRunManager.cpp:158
    #9  0xb4e3b64f in GiGa::retrieveTheEvent (this=0xc480220, event=@0xbfa6c9d8) at ../src/component/GiGa.cpp:469
    #10 0xb4e38564 in GiGa::operator>> (this=0xc480220, event=@0xbfa6c9d8) at ../src/component/GiGaIGiGaSvc.cpp:73
    #11 0xb4e362fa in GiGa::retrieveEvent (this=0xc480220, event=@0xbfa6c9d8) at ../src/component/GiGaIGiGaSvc.cpp:211
    #12 0xb507fcd3 in DsPullEvent::execute (this=0xc473470) at ../src/DsPullEvent.cc:54
    #13 0x046d6408 in Algorithm::sysExecute (this=0xc473470) at ../src/Lib/Algorithm.cpp:558
    #14 0x03a61d4e in DybBaseAlg::sysExecute (this=0xc473470) at ../src/lib/DybBaseAlg.cc:53
    #15 0x01cf0fd4 in GaudiSequencer::execute (this=0xbf36020) at ../src/lib/GaudiSequencer.cpp:100
    #16 0x046d6408 in Algorithm::sysExecute (this=0xbf36020) at ../src/Lib/Algorithm.cpp:558
    #17 0x01c8868f in GaudiAlgorithm::sysExecute (this=0xbf36020) at ../src/lib/GaudiAlgorithm.cpp:161
    #18 0x0475241a in MinimalEventLoopMgr::executeEvent (this=0xbaf2f98) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #19 0x03b20956 in DybEventLoopMgr::executeEvent (this=0xbaf2f98, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #20 0x03b2118a in DybEventLoopMgr::nextEvent (this=0xbaf2f98, maxevt=10) at ../src/DybEventLoopMgr.cpp:188
    #21 0x04750dbd in MinimalEventLoopMgr::executeRun (this=0xbaf2f98, maxevt=10) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #22 0x08c086d9 in ApplicationMgr::executeRun (this=0xb7b9ad0, evtmax=10) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #23 0x0239af57 in method_3426 (retaddr=0xc5821b0, o=0xb7b9efc, arg=@0xb825c50) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #24 0x0030cadd in ROOT::Cintex::Method_stub_with_context (context=0xb825c48, result=0xc5cafe4, libp=0xc5cb03c) at cint/cintex/src/CINTFunctional.cxx:319



G4VProcess::PostStepGPIL
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    (gdb) frame 1
    #1  0x07247e95 in G4VProcess::PostStepGPIL (this=0xd37b178, track=@0x10c8cd90, previousStepSize=0, condition=0xc481da0) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:464
    464        =PostStepGetPhysicalInteractionLength(track, previousStepSize, condition);
    (gdb) list
    459     inline G4double G4VProcess::PostStepGPIL( const G4Track& track,
    460                                        G4double   previousStepSize,
    461                                        G4ForceCondition* condition )
    462     {
    463       G4double value
    464        =PostStepGetPhysicalInteractionLength(track, previousStepSize, condition);
    465       return thePILfactor*value;
    466     }


G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    131 inline G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
    132                              const G4Track& track,
    133                  G4double   previousStepSize,
    134                  G4ForceCondition* condition
    135                 )
    136 {
    137   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
    138     // beggining of tracking (or just after DoIt of this process)
    139     ResetNumberOfInteractionLengthLeft();
    140   } else if ( previousStepSize > 0.0) {
    141     // subtract NumberOfInteractionLengthLeft 
    142     SubtractNumberOfInteractionLengthLeft(previousStepSize);
    143   } else {
    144     // zero step
    145     //  DO NOTHING
    146   }
    147 
    148   // condition is set to "Not Forced"
    149   *condition = NotForced;
    150 
    151   // get mean free path
    152   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    153 
    154   G4double value;
    155   if (currentInteractionLength <DBL_MAX) {
    156     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
    157   } else {
    158     value = DBL_MAX;
    159   }
    160 #ifdef G4VERBOSE
    161   if (verboseLevel>1){
    162     G4cout << "G4VDiscreteProcess::PostStepGetPhysicalInteractionLength ";
    163     G4cout << "[ " << GetProcessName() << "]" <<G4endl;
    164     track.GetDynamicParticle()->DumpInfo();
    165     G4cout << " in Material  " <<  track.GetMaterial()->GetName() <<G4endl;
    166     G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
    167   }
    168 #endif
    169   return value;
    170 }



processes/management/include/G4VProcess.hh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    076 class G4VProcess
    077 {
    078   //  A virtual class for physics process objects. It defines
    079   //  public methods which describe the behavior of a
    080   //  physics process.
    081 
    ...
    147       virtual G4double PostStepGetPhysicalInteractionLength(
    148                              const G4Track& track,
    149                  G4double   previousStepSize,
    150                  G4ForceCondition* condition
    151                 ) = 0;
    152 
    153       //  Returns the Step-size (actual length) which is allowed 
    154       //  by "this" process. (for AtRestGetPhysicalInteractionLength,
    155       //  return value is Step-time) The NumberOfInteractionLengthLeft is
    156       //  recalculated by using previousStepSize and the Step-size is 
    157       //  calucalted accoding to the resultant NumberOfInteractionLengthLeft.
    158       //  using NumberOfInteractionLengthLeft, which is recalculated at 
    159       //    arguments
    160       //      const G4Track&    track:
    161       //        reference to the current G4Track information
    162       //      G4double*          previousStepSize: 
    163       //        the Step-size (actual length) of the previous Step 
    164       //        of this track. Negative calue indicates that
    165       //        NumberOfInteractionLengthLeft must be reset.
    166       //        the current physical interaction legth of this process
    167       //      G4ForceCondition* condition:
    168       //        the flag indicates DoIt of this process is forced 
    169       //        to be called
    170       //         Forced:    Corresponding DoIt is forced
    171       //         NotForced: Corresponding DoIt is called 
    172       //                    if the Step size of this Step is determined 
    173       //                    by this process
    174       //        !! AlongStepDoIt is always called !! 
    175       //      G4double& currentMinimumStep:
    176       //        this value is used for transformation of
    177       //        true path length to geometrical path length
    178 
    179       G4double GetCurrentInteractionLength() const;
    180       // Returns currentInteractionLength
    181 
    182       ////////// PIL factor ////////
    183       void SetPILfactor(G4double value);
    184       G4double GetPILfactor() const;
    185       // Set/Get factor for PhysicsInteractionLength 
    186       // which is passed to G4SteppingManager for both AtRest and PostStep
    187 
    188       // These three GPIL methods are used by Stepping Manager.
    189       // They invoke virtual GPIL methods listed above.
    190       // As for AtRest and PostStep the returned value is multipled by thePILfactor 
    191       // 
    ...
    287   protected:
    288       G4VParticleChange* pParticleChange;
    289       //  The pointer to G4VParticleChange object 
    290       //  which is modified and returned by address by the DoIt() method.
    291       //  This pointer should be set in each physics process
    292       //  after construction of derived class object.  
    293 
    294       G4ParticleChange aParticleChange;
    295       //  This object is kept for compatibility with old scheme
    296       //  This will be removed in future
    297 
    298       G4double          theNumberOfInteractionLengthLeft;
    299      // The flight length left for the current tracking particle
    300      // in unit of "Interaction length".
    301 
    302       G4double          currentInteractionLength;
    303      // The InteractionLength in the current material
    304 
    305  public: // with description
    306       virtual void      ResetNumberOfInteractionLengthLeft();
    307      // reset (determine the value of)NumberOfInteractionLengthLeft
    308 
    309  protected:  // with description
    310      virtual void      SubtractNumberOfInteractionLengthLeft(
    311                   G4double previousStepSize
    312                                 );
    313      // subtract NumberOfInteractionLengthLeft by the value corresponding to 
    314      // previousStepSize      



Rayleigh Scattering
~~~~~~~~~~~~~~~~~~~~

The differential cross section in Rayleigh scattering, `sigma/omega` , is proportional to
`cos2(theta)`, where `theta` is the polar angle of the new polarization vector with respect to
the old polarization vector. The `G4OpRayleigh` scattering process samples this
angle accordingly and then calculates the scattered photon's new direction by
requiring that it be perpendicular to the photon's new polarization in such a
way that the final direction, initial and final polarizations are all in one
plane. This process thus depends on the particle's polarization (spin). The
photon's polarization is a data member of the `G4DynamicParticle` class.

A photon which is not assigned a polarization at production, either via the
`SetPolarization` method of the `G4PrimaryParticle` class, or indirectly with the
`SetParticlePolarization` method of the `G4ParticleGun` class, may not be Rayleigh
scattered. Optical photons produced by the `G4Cerenkov` process have inherently a
polarization perpendicular to the cone's surface at production. Scintillation
photons have a random linear polarization perpendicular to their direction.

The process requires a `G4MaterialPropertiesTable` to be filled by the user with
Rayleigh scattering length data. The Rayleigh scattering attenuation length is
the average distance traveled by a photon before it is Rayleigh scattered in
the medium and it is the distance returned by the `GetMeanFreePath` method. The
`G4OpRayleigh` class provides a `RayleighAttenuationLengthGenerator` method which
calculates the attenuation coefficient of a medium following the
Einstein-Smoluchowski formula whose derivation requires the use of statistical
mechanics, includes temperature, and depends on the isothermal compressibility
of the medium. This generator is convenient when the Rayleigh attenuation
length is not known from measurement but may be calculated from first
principles using the above material constants. For a medium named Water and no
Rayleigh scattering attenutation length specified by the user, the program
automatically calls the RayleighAttenuationLengthGenerator
which calculates it for 10 degrees Celsius liquid water.







G4SteppingManager::GetProcessNumber
--------------------------------------

Three categories of DoIt:

`AtRestDoIt`
      eg decays       
`AlongStepDoIt`
      relevant for `G4VContinuousProcess`
`PostStepDoIt`
      relevant for `G4VDiscreteProcess` like those that optical photons undergo: Absorption, Scattering, Boundaries


::

     56 /////////////////////////////////////////////////
     57 void G4SteppingManager::GetProcessNumber()
     58 /////////////////////////////////////////////////
     59 {
     60 #ifdef debug
     61   G4cout<<"G4SteppingManager::GetProcessNumber: is called track="<<fTrack<<G4endl;
     62 #endif
     63 
     64   G4ProcessManager* pm= fTrack->GetDefinition()->GetProcessManager();
     65         if(!pm)
     66   {
     67     G4cout<<"G4SteppingManager::GetProcessNumber: ProcessManager=0 for particle="
     68           <<fTrack->GetDefinition()->GetParticleName()<<", PDG_code="
     69           <<fTrack->GetDefinition()->GetPDGEncoding()<<G4endl;
     70                 G4Exception("G4SteppingManager::GetProcessNumber: Process Manager is not found.");
     71   }
     72 
     73 // AtRestDoits
     74    MAXofAtRestLoops =        pm->GetAtRestProcessVector()->entries();
     75    fAtRestDoItVector =       pm->GetAtRestProcessVector(typeDoIt);
     76    fAtRestGetPhysIntVector = pm->GetAtRestProcessVector(typeGPIL);
     77 #ifdef debug
     78   G4cout<<"G4SteppingManager::GetProcessNumber: #ofAtRest="<<MAXofAtRestLoops<<G4endl;
     79 #endif
     80 
     81 // AlongStepDoits
     82    MAXofAlongStepLoops = pm->GetAlongStepProcessVector()->entries();
     83    fAlongStepDoItVector = pm->GetAlongStepProcessVector(typeDoIt);
     84    fAlongStepGetPhysIntVector = pm->GetAlongStepProcessVector(typeGPIL);
     85 #ifdef debug
     86             G4cout<<"G4SteppingManager::GetProcessNumber:#ofAlongStp="<<MAXofAlongStepLoops<<G4endl;
     87 #endif
     88 
     89 // PostStepDoits
     90    MAXofPostStepLoops = pm->GetPostStepProcessVector()->entries();
     91    fPostStepDoItVector = pm->GetPostStepProcessVector(typeDoIt);
     92    fPostStepGetPhysIntVector = pm->GetPostStepProcessVector(typeGPIL);
     93 #ifdef debug
     94             G4cout<<"G4SteppingManager::GetProcessNumber: #ofPostStep="<<MAXofPostStepLoops<<G4endl;
     95 #endif
     96 
     97    if (SizeOfSelectedDoItVector<MAXofAtRestLoops    ||
     98        SizeOfSelectedDoItVector<MAXofAlongStepLoops ||
     99        SizeOfSelectedDoItVector<MAXofPostStepLoops  )
     100             {
     101               G4cout<<"G4SteppingManager::GetProcessNumber: SizeOfSelectedDoItVector="
     102            <<SizeOfSelectedDoItVector<<" is smaller then one of MAXofAtRestLoops="
     103            <<MAXofAtRestLoops<<" or MAXofAlongStepLoops="<<MAXofAlongStepLoops
     104            <<" or MAXofPostStepLoops="<<MAXofPostStepLoops<<G4endl;
     105                     G4Exception("G4SteppingManager::GetProcessNumber: The array size is smaller than the actutal number of processes. Chnage G4SteppingManager.hh and recompile is needed.");
     106    }
     107 }


How are the relevant processes determined ?
----------------------------------------------

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H fPostStepDoItVector {} \;
    ./tracking/src/G4SteppingManager2.cc:   fPostStepDoItVector = pm->GetPostStepProcessVector(typeDoIt);
    ./tracking/src/G4SteppingManager2.cc:         fCurrentProcess = (*fPostStepDoItVector)[np];
    ./tracking/src/G4SteppingVerbose.cc:             ptProcManager = (*fPostStepDoItVector)[np];
    ./tracking/src/G4SteppingVerbose.cc:             ptProcManager = (*fPostStepDoItVector)[np];
    ./tracking/src/G4VSteppingVerbose.cc:   fPostStepDoItVector = fManager->GetfPostStepDoItVector();
    [blyth@cms01 source]$ 





What distribution is used for OP times, energy 
----------------------------------------------------

DsPmtSensDet::ProcessHits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the G4Step, energies and times feed into creating hits. 

For OP, wavelength is more relevant than energy. 
From http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/TrackingAndPhysics/physicsProcess.html

   * Optical photons are generated in GEANT4 without energy conservation and
     their energy must therefore not be tallied as part of the energy balance of an event.



::

    318 bool DsPmtSensDet::ProcessHits(G4Step* step,
    319                                G4TouchableHistory* /*history*/)
    320 {
    321     //if (!step) return false; just crash for now if not defined
    322 
    323     // Find out what detector we are in (ADx, IWS or OWS)
    324     G4StepPoint* preStepPoint = step->GetPreStepPoint();
    325 
    326     double energyDep = step->GetTotalEnergyDeposit();
    ...
    ...
    ...
    434     double wavelength = CLHEP::twopi*CLHEP::hbarc/energyDep;
    ...
    ...
    ...
    459     DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
    460 
    461     // base hit
    462 
    463     // Time since event created
    464     sphit->setHitTime(preStepPoint->GetGlobalTime());
    465 
    466     //#include "G4NavigationHistory.hh"
    467 
    468     const G4AffineTransform& trans = hist->GetHistory()->GetTopTransform();
    469     const G4ThreeVector& global_pos = preStepPoint->GetPosition();
    470     G4ThreeVector pos = trans.TransformPoint(global_pos);
    471     sphit->setLocalPos(pos);
    472     sphit->setSensDetId(pmtid);
    473    
    474     // pmt hit
    475     // sphit->setDir(...);       // for now
    476     G4ThreeVector pol = trans.TransformAxis(track->GetPolarization());
    477     pol = pol.unit();
    478     G4ThreeVector dir = trans.TransformAxis(track->GetMomentum());
    479     dir = dir.unit();
    480     sphit->setPol(pol);
    481     sphit->setDir(dir);
    482     sphit->setWavelength(wavelength);
    483     sphit->setType(0);
    484     // G4cerr<<"PMT: set hit weight "<<weight<<G4endl; //gonchar
    485     sphit->setWeight(weight);


Where do the times come from ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H SetGlobalTime {} \;
    ./track/src/G4ParticleChangeForDecay.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    ./track/src/G4ParticleChange.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    ./track/src/G4ParticleChange.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    ./processes/hadronic/models/lll_fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(getnage_(&i)*second);
    ./processes/hadronic/models/lll_fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(getpage_(&i)*second);
    ./processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    ./processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    [blyth@cms01 source]$ 

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H ProposeGlobalTime {} \;
    ./processes/hadronic/models/radioactive_decay/src/G4RadioactiveDecay.cc:      fParticleChangeForRadDecay.ProposeGlobalTime( finalGlobalTime );
    ./processes/transportation/src/G4Transportation.cc:  fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    ./processes/transportation/src/G4CoupledTransportation.cc:  fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    ./processes/decay/src/G4UnknownDecay.cc:  fParticleChangeForDecay.ProposeGlobalTime( finalGlobalTime );
    ./processes/decay/src/G4Decay.cc:  fParticleChangeForDecay.ProposeGlobalTime( finalGlobalTime );
    [blyth@cms01 source]$ 



G4Transportation::AlongStepDoIt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For OP, what determines the GlobalTime is

* `startTime + (stepLength/finalVelocity)`  

So question becomes: where is stepLength distribution implemented ? Each process provides a MeanFreePath, where 
is the dice rolled ? 

::

    450 G4VParticleChange* G4Transportation::AlongStepDoIt( const G4Track& track,
    451                                                     const G4Step&  stepData )
    452 {
    453   static G4int noCalls=0;
    454   static const G4ParticleDefinition* fOpticalPhoton =
    455            G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");
    456 
    457   noCalls++;
    458 
    459   fParticleChange.Initialize(track) ;
    460 
    461   //  Code for specific process 
    462   //
    463   fParticleChange.ProposePosition(fTransportEndPosition) ;
    464   fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir) ;
    465   fParticleChange.ProposeEnergy(fTransportEndKineticEnergy) ;
    466   fParticleChange.SetMomentumChanged(fMomentumChanged) ;
    467 
    468   fParticleChange.ProposePolarization(fTransportEndSpin);
    469 
    470   G4double deltaTime = 0.0 ;
    471 
    472   // Calculate  Lab Time of Flight (ONLY if field Equations used it!)
    473      // G4double endTime   = fCandidateEndGlobalTime;
    474      // G4double delta_time = endTime - startTime;
    475 
    476   G4double startTime = track.GetGlobalTime() ;
    477 
    478   if (!fEndGlobalTimeComputed)
    479   {
    480      // The time was not integrated .. make the best estimate possible
    481      //
    482      G4double finalVelocity   = track.GetVelocity() ;
    483      G4double initialVelocity = stepData.GetPreStepPoint()->GetVelocity() ;
    484      G4double stepLength      = track.GetStepLength() ;
    485 
    486      deltaTime= 0.0;  // in case initialVelocity = 0 
    487      const G4DynamicParticle* fpDynamicParticle = track.GetDynamicParticle();
    488      if (fpDynamicParticle->GetDefinition()== fOpticalPhoton)
    489      {
    490         //  A photon is in the medium of the final point
    491         //  during the step, so it has the final velocity.
    492         deltaTime = stepLength/finalVelocity ;
    493      }
    494      else if (finalVelocity > 0.0)
    495      {
    496         G4double meanInverseVelocity ;
    497         // deltaTime = stepLength/finalVelocity ;
    498         meanInverseVelocity = 0.5
    499                             * ( 1.0 / initialVelocity + 1.0 / finalVelocity ) ;
    500         deltaTime = stepLength * meanInverseVelocity ;
    501      }
    502      else if( initialVelocity > 0.0 )
    503      {
    504         deltaTime = stepLength/initialVelocity ;
    505      }
    506      fCandidateEndGlobalTime   = startTime + deltaTime ;
    507   }
    508   else
    509   {
    510      deltaTime = fCandidateEndGlobalTime - startTime ;
    511   }
    512 
    513   fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;



G4SteppingManager::DefinePhysicalStepLength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H DefinePhysicalStepLength {} \;
    ./tracking/src/G4SteppingManager.cc:     DefinePhysicalStepLength();
    ./tracking/src/G4SteppingManager2.cc: void G4SteppingManager::DefinePhysicalStepLength()
    ./tracking/src/G4SteppingManager2.cc:} // void G4SteppingManager::DefinePhysicalStepLength() //
    ./tracking/src/G4SteppingVerbose.cc:    G4cout << G4endl << " >>DefinePhysicalStepLength (List of proposed StepLengths): "  << G4endl;




::

    118  void G4SteppingManager::DefinePhysicalStepLength()
    119 /////////////////////////////////////////////////////////
    120 {
    121 
    122 // ReSet the counter etc.
    123    PhysicalStep  = DBL_MAX;          // Initialize by a huge number    
    124    physIntLength = DBL_MAX;          // Initialize by a huge number    
    125 #ifdef G4VERBOSE
    126                          // !!!!! Verbose
    127            if(verboseLevel>0) fVerbose->DPSLStarted();
    128 #endif
    129 
    130 // Obtain the user defined maximum allowed Step in the volume
    131 //   1997.12.13 adds argument for  GetMaxAllowedStep by K.Kurashige
    132 //   2004.01.20 This block will be removed by Geant4 7.0 
    133 //   G4UserLimits* ul= fCurrentVolume->GetLogicalVolume()->GetUserLimits();
    134 //   if (ul) {
    135 //      physIntLength = ul->GetMaxAllowedStep(*fTrack);
    136 //#ifdef G4VERBOSE
    137 //                         // !!!!! Verbose
    138 //           if(verboseLevel>0) fVerbose->DPSLUserLimit();
    139 //#endif
    140 //   }
    141 //
    142 //   if(physIntLength < PhysicalStep ){
    143 //      PhysicalStep = physIntLength;
    144 //      fStepStatus = fUserDefinedLimit;
    145 //      fStep->GetPostStepPoint()
    146 //           ->SetProcessDefinedStep(NULL);
    147 //      // Take note that the process pointer is 'NULL' if the Step
    148 //      // is defined by the user defined limit.
    149 //   }
    150 //   2004.01.20 This block will be removed by Geant4 7.0 
    151 
    152 // GPIL for PostStep
    153    fPostStepDoItProcTriggered = MAXofPostStepLoops;
    154 
    155    for(size_t np=0; np < MAXofPostStepLoops; np++){
    156      fCurrentProcess = (*fPostStepGetPhysIntVector)(np);
    157      if (fCurrentProcess== NULL) {
    158        (*fSelectedPostStepDoItVector)[np] = InActivated;
    159        continue;
    160      }   // NULL means the process is inactivated by a user on fly.
    161 
    162      physIntLength = fCurrentProcess->
    163                      PostStepGPIL( *fTrack,
    164                                                  fPreviousStepSize,
    165                                                       &fCondition );
    166 #ifdef G4VERBOSE
    167                          // !!!!! Verbose
    168            if(verboseLevel>0) fVerbose->DPSLPostStep();
    169 #endif
    170 
    171      switch (fCondition) {
    172      case ExclusivelyForced:
    173          (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
    174          fStepStatus = fExclusivelyForcedProc;
    175          fStep->GetPostStepPoint()
    176          ->SetProcessDefinedStep(fCurrentProcess);
    177          break;
    178      case Conditionally:
    179          (*fSelectedPostStepDoItVector)[np] = Conditionally;
    180          break;
    181      case Forced:





G4SteppingManager::DefinePhysicalStepLength
----------------------------------------------

Only 6 processes ?
~~~~~~~~~~~~~~~~~~~~~

::

    (gdb) p fCurrentProcess->GetProcessName()
    $9 = (const G4String &) @0xc094a20: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xc094c04 "Transportation"}}, <No data fields>}
    (gdb) c
    Continuing.
        1 -1.43e+04   -8e+05 -1.14e+03  2.31e-06        0  3.3e+03   3.3e+03 /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof Transportation

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $10 = (const G4String &) @0xce5a190: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xc484aec "Scintillation"}}, <No data fields>}
    (gdb) c
    Continuing.

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $11 = (const G4String &) @0xd37bc80: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xd379024 "fast_sim_man"}}, <No data fields>}
    (gdb) c
    Continuing.

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $12 = (const G4String &) @0xd37b258: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xd37b164 "OpBoundary"}}, <No data fields>}
    (gdb) c
    Continuing.

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $13 = (const G4String &) @0xd3782f8: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xce5eb2c "OpRayleigh"}}, <No data fields>}
    (gdb) c
    Continuing.

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $14 = (const G4String &) @0xd3779e8: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xce6044c "OpAbsorption"}}, <No data fields>}
    (gdb) c
    Continuing.

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();
    (gdb) p fCurrentProcess->GetProcessName()
    $15 = (const G4String &) @0xc094a20: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xc094c04 "Transportation"}}, <No data fields>}
    (gdb) c
    Continuing.
        2 -1.45e+04   -8e+05 -1.31e+03  2.31e-06        0      208  3.51e+03 /dd/Geometry/RPC/lvNearRPCRoof#pvNearUnSlopModArray#pvNearUnSlopModOne:3#pvNearUnSlopMod:2#pvNearSlopModUnit Transportation
    Step#    X(mm)    Y(mm)    Z(mm) KinE(MeV)  dE(MeV) StepLeng TrackLeng  NextVolume ProcName
        0 -1.23e+04   -8e+05 1.56e+03  5.77e-06        0        0         0 /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop initStep

    Breakpoint 2, G4SteppingManager::DefinePhysicalStepLength (this=0xc481c98) at src/G4SteppingManager2.cc:168
    168                if(verboseLevel>0) fVerbose->DPSLPostStep();



2.4.4.  Interaction with Physics Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForToolkitDeveloper/html/ch02s04.html

The interaction of the tracking category with the physics processes is done in
two ways. First each process can limit the step length through one of its three
`GetPhysicalInteractionLength()` methods, `AtRest`, `AlongStep`, or `PostStep`. 
Second, for the selected processes the `DoIt` (`AtRest`, `AlongStep` or `PostStep`) methods are
invoked. All this interaction is managed by the `Stepping` method of
`G4SteppingManager`. To calculate the step length, the `DefinePhysicalStepLength()`
method is called. The flow of this method is the following:

Obtain maximum allowed Step in the volume define by the user through G4UserLimits.

#. The `PostStepGetPhysicalInteractionLength` of all active processes is called. 
#. Each process returns a step length and the minimum one is chosen. 
#. This method also returns a `G4ForceCondition` flag, to indicate if the process is forced or not: 

`Forced`
       Corresponding `PostStepDoIt` is forced. 
`NotForced` 
       Corresponding `PostStepDoIt` is not forced unless this process limits the step. 
`Conditionally` 
       Only when `AlongStepDoIt` limits the step, corresponding `PoststepDoIt` is invoked. 
`ExclusivelyForced` 
       Corresponding `PostStepDoIt` is exclusively forced. All other `DoIt` including `AlongStepDoIts` are ignored.

The `AlongStepGetPhysicalInteractionLength` method of all active processes is
called. Each process returns a step length and the minimum of these and the
This method also returns a `fGPILSelection` flag, to indicate if the process is
the selected one can be is forced or not. 

`CandidateForSelection`
    this process can be the winner. If its step length is the smallest, it will be the process
    defining the step (the process = 
`NotCandidateForSelection`
    this process cannot be the winner. Even if its step length is taken as the smallest, it will not be
    the process defining the step




G4 Stepping, Process mechanics
------------------------------------

* https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch05.html


What is a Process?
~~~~~~~~~~~~~~~~~~~

Only processes can change information of G4Track and add secondary tracks via
ParticleChange. G4VProcess is a base class of all processes and it has 3 kinds
of DoIt and GetPhysicalInteraction methods in order to describe interactions
generically. If a user want to modify information of G4Track, he (or she)
SHOULD create a special process for the purpose and register the process to the
particle.

What is a ParticleChange?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Processes do NOT change any information of G4Track directly in their DoIt.
Instead, they proposes changes as a result of interactions by using
ParticleChange. After each DoIt, ParticleChange updates PostStepPoint based on
proposed changes. Then, G4Track is updated after finishing all AlongStepDoIts
and after each PostStepDoIt.





