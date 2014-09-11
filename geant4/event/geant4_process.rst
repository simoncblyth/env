Geant4 Process
================

.. contents:: :local:

Overview
----------

Trace how Geant4 processes work, with a view to glueing 
external batched optical photon propagation into geant4 via 
a pseudo process. 


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

External Shoe-horning 
~~~~~~~~~~~~~~~~~~~~~~~~

Begs question:

#. how to force only my process takes control, and dont waste CPU on other processes ? 
 
   * create a special particle type *ExternalOpticalPhoton* and switch standard optical photon type 
     for it in all the Optical Photon creating processes  





Process Example : G4OpRayleigh
--------------------------------


`geant4.10.00.p01/source/processes/optical/include/G4OpRayleigh.hh`::

    076 class G4OpRayleigh : public G4VDiscreteProcess
    077 { 
    ...
    099 public:
    100 
    ...
    105         G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
    106         // Returns true -> 'is applicable' only for an optical photon.
    107 
    108         void BuildPhysicsTable(const G4ParticleDefinition& aParticleType);
    109         // Build table at a right time
    110 
    111         G4double GetMeanFreePath(const G4Track& aTrack,
    112                  G4double ,
    113                                  G4ForceCondition* );
    114         // Returns the mean free path for Rayleigh scattering in water.
    115         // --- Not yet implemented for other materials! ---
    116 
    117         G4VParticleChange* PostStepDoIt(const G4Track& aTrack,
    118                                        const G4Step&  aStep);
    119         // This is the method implementing Rayleigh scattering.
    120 
    121         G4PhysicsTable* GetPhysicsTable() const;
    122         // Returns the address of the physics table.
    123 
    124         void DumpPhysicsTable() const;
    125         // Prints the physics table.
    126 



Where are the canonical calls to GetMeanFreePath
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ordering of MeanFreePath of applicable processes is used to decide which process
to invoke.

::

    delta:source blyth$ find . -name '*.cc' -exec grep -H \ GetMeanFreePath\( {} \;
    ./processes/decay/src/G4Decay.cc:    currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    ./processes/management/src/G4VContinuousDiscreteProcess.cc:  currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    ./processes/management/src/G4VDiscreteProcess.cc:  currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    ./processes/management/src/G4VRestContinuousDiscreteProcess.cc:  currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    ./processes/management/src/G4VRestDiscreteProcess.cc:  currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
    delta:source blyth$ 


::

     71 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
     72                              const G4Track& track,
     73                  G4double   previousStepSize,
     74                  G4ForceCondition* condition
     75                 )
     76 {
     77   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
     78     // beggining of tracking (or just after DoIt of this process)
     79     ResetNumberOfInteractionLengthLeft();
     80   } else if ( previousStepSize > 0.0) {
     81     // subtract NumberOfInteractionLengthLeft 
     82     SubtractNumberOfInteractionLengthLeft(previousStepSize);
     83   } else {
     84     // zero step
     85     //  DO NOTHING
     86   }
     87 
     88   // condition is set to "Not Forced"
     89   *condition = NotForced;
     90 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93 
     94   G4double value;
     95   if (currentInteractionLength <DBL_MAX) {
     96     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
     97   } else {
     98     value = DBL_MAX;
     99   }
     ..
     09   return value;
     10 }



G4ProcessManager 
------------------

canonical PostStepGetPhysicalInteractionLength call `processes/management/include/G4ProcessManager.hh`::

    035 //   ----------------  G4ProcessManager  -----------------
    036 // Class Description 
    037 //  It collects all physics a particle can undertake as seven vectors.
    038 //  These vectors are 
    039 //   one vector for all processes (called as "process List")
    040 //   two vectors for processes with AtRestGetPhysicalInteractionLength
    041 //                                    and AtRestDoIt
    042 //   two vectors for processes with AlongStepGetPhysicalInteractionLength
    043 //                                    and AlongStepDoIt
    044 //   two vectors for processes with PostStepGetPhysicalInteractionLength
    045 //                                    and PostStepDoIt
    046 //  The tracking will message three types of GetPhysicalInteractionLength
    047 //  in order to limit the Step and select the occurence of processes. 
    048 //  It will message the corresponding DoIt() to apply the selected 
    049 //  processes. In addition, the Tracking will limit the Step
    050 //  and select the occurence of the processes according to
    051 //  the shortest physical interaction length computed (except for
    052 //  processes at rest, for which the Tracking will select the
    053 //  occurence of the process which returns the shortest mean
    054 //  life-time from the GetPhysicalInteractionLength()).
    ...
    160       G4ProcessVector* GetPostStepProcessVector(
    161                    G4ProcessVectorTypeIndex typ = typeGPIL
    162                               ) const;
    163       //  Returns the address of the vector of processes for
    164       //    PostStepGetPhysicalInteractionLength      idx =0
    165       //    PostStepGetPhysicalDoIt                   idx =1



G4SteppingManager
------------------

Collect process vectors for track particle type `tracking/src/G4SteppingManager2.cc`::

     56 void G4SteppingManager::GetProcessNumber()
     57 /////////////////////////////////////////////////
     58 {
     63 
     64   G4ProcessManager* pm= fTrack->GetDefinition()->GetProcessManager();
     ..
     76 // AtRestDoits
     77    MAXofAtRestLoops =        pm->GetAtRestProcessVector()->entries();
     78    fAtRestDoItVector =       pm->GetAtRestProcessVector(typeDoIt);
     79    fAtRestGetPhysIntVector = pm->GetAtRestProcessVector(typeGPIL);
     .. 
     85 // AlongStepDoits
     86    MAXofAlongStepLoops = pm->GetAlongStepProcessVector()->entries();
     87    fAlongStepDoItVector = pm->GetAlongStepProcessVector(typeDoIt);
     88    fAlongStepGetPhysIntVector = pm->GetAlongStepProcessVector(typeGPIL);
     .. 
     94 // PostStepDoits
     95    MAXofPostStepLoops = pm->GetPostStepProcessVector()->entries();
     96    fPostStepDoItVector = pm->GetPostStepProcessVector(typeDoIt);
     97    fPostStepGetPhysIntVector = pm->GetPostStepProcessVector(typeGPIL);


ExclusivelyForced maybe way to restrict to just one process. 
Nope, cannot restrict to one process:  need to arrange the ordering such that the the processes 
that generate Optical Photons (Cherenkov, Scintillaton) go first and the chain is stopped at the pseudo process.

Hmm its kinda a replacement for the tail transport process.  Sorta but cannot allow the 
optical processes to do their thing. 


::

    128  void G4SteppingManager::DefinePhysicalStepLength()
    130 {
    ...
    162 // GPIL for PostStep
    163    fPostStepDoItProcTriggered = MAXofPostStepLoops;
    164 
    165    for(size_t np=0; np < MAXofPostStepLoops; np++){
    166      fCurrentProcess = (*fPostStepGetPhysIntVector)(np);
    ...
    172      physIntLength = fCurrentProcess->
    173                      PostStepGPIL( *fTrack,
    174                                                  fPreviousStepSize,
    175                                                       &fCondition );
    ...
    181      switch (fCondition) {
    182      case ExclusivelyForced:
    183          (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
    184          fStepStatus = fExclusivelyForcedProc;
    185          fStep->GetPostStepPoint()
    186          ->SetProcessDefinedStep(fCurrentProcess);
    187          break;
    188      case Conditionally:
    189        //        (*fSelectedPostStepDoItVector)[np] = Conditionally;
    190          G4Exception("G4SteppingManager::DefinePhysicalStepLength()", "Tracking1001", FatalException, "This feature no more supported");
    191 
    192          break;
    193      case Forced:
    194          (*fSelectedPostStepDoItVector)[np] = Forced;
    195          break;
    196      case StronglyForced:
    197          (*fSelectedPostStepDoItVector)[np] = StronglyForced;
    198          break;
    199      default:
    200          (*fSelectedPostStepDoItVector)[np] = InActivated;
    201          break;
    202      }
    203 
    204 
    205 
    206      if (fCondition==ExclusivelyForced) {
    207      for(size_t nrest=np+1; nrest < MAXofPostStepLoops; nrest++){
    208          (*fSelectedPostStepDoItVector)[nrest] = InActivated;
    209      }
    210      return;  // Take note the 'return' at here !!! 
    211      }
    212      else{
    213      if(physIntLength < PhysicalStep ){
    214          PhysicalStep = physIntLength;
    215          fStepStatus = fPostStepDoItProc;
    216          fPostStepDoItProcTriggered = G4int(np);
    217          fStep->GetPostStepPoint()
    218          ->SetProcessDefinedStep(fCurrentProcess);
    219      }
    220      }
    221 
    222 
    223    }


    483 void G4SteppingManager::InvokePostStepDoItProcs()
    484 ////////////////////////////////////////////////////////
    485 {
    486 
    487 // Invoke the specified discrete processes
    488    for(size_t np=0; np < MAXofPostStepLoops; np++){
    489    //
    490    // Note: DoItVector has inverse order against GetPhysIntVector
    491    //       and SelectedPostStepDoItVector.
    492    //
    493      G4int Cond = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np-1];
    494      if(Cond != InActivated){
    495        if( ((Cond == NotForced) && (fStepStatus == fPostStepDoItProc)) ||
    496        ((Cond == Forced) && (fStepStatus != fExclusivelyForcedProc)) ||
    497        //      ((Cond == Conditionally) && (fStepStatus == fAlongStepDoItProc)) ||
    498        ((Cond == ExclusivelyForced) && (fStepStatus == fExclusivelyForcedProc)) ||
    499        ((Cond == StronglyForced) )
    500       ) {
    501 
    502      InvokePSDIP(np);
    503          if ((np==0) && (fTrack->GetNextVolume() == 0)){
    504            fStepStatus = fWorldBoundary;
    505            fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    506          }
    507        }
    508      } //if(*fSelectedPostStepDoItVector(np)........
    509 
    510      // Exit from PostStepLoop if the track has been killed,
    511      // but extra treatment for processes with Strongly Forced flag
    512      if(fTrack->GetTrackStatus() == fStopAndKill) {
    513        for(size_t np1=np+1; np1 < MAXofPostStepLoops; np1++){
    514      G4int Cond2 = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np1-1];
    515      if (Cond2 == StronglyForced) {
    516        InvokePSDIP(np1);
    517          }
    518        }
    519        break;
    520      }
    521    } //for(size_t np=0; np < MAXofPostStepLoops; np++){
    522 }

    526 void G4SteppingManager::InvokePSDIP(size_t np)
    527 {
    528          fCurrentProcess = (*fPostStepDoItVector)[np];
    529          fParticleChange
    530             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
    531 
    532          // Update PostStepPoint of Step according to ParticleChange
    533      fParticleChange->UpdateStepForPostStep(fStep);
    534 #ifdef G4VERBOSE
    535                  // !!!!! Verbose
    536            if(verboseLevel>0) fVerbose->PostStepDoItOneByOne();
    537 #endif
    538          // Update G4Track according to ParticleChange after each PostStepDoIt
    539          fStep->UpdateTrack();
    540 
    541          // Update safety after each invocation of PostStepDoIts
    542          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    543 
    544          // Now Store the secondaries from ParticleChange to SecondaryList
    545          G4Track* tempSecondaryTrack;
    546          G4int    num2ndaries;
    547 
    548          num2ndaries = fParticleChange->GetNumberOfSecondaries();
    549 
    550          for(G4int DSecLoop=0 ; DSecLoop< num2ndaries; DSecLoop++){
    551             tempSecondaryTrack = fParticleChange->GetSecondary(DSecLoop);
    ...
    579          } //end of loop on secondary 
    580 
    581          // Set the track status according to what the process defined
    582          fTrack->SetTrackStatus( fParticleChange->GetTrackStatus() );
    583 
    584          // clear ParticleChange
    585          fParticleChange->Clear();
    586 }



G4TrackingManager
-------------------

`tracking/src/G4TrackingManager.cc`::

     48 G4TrackingManager::G4TrackingManager()
     49 //////////////////////////////////////
     50   : fpUserTrackingAction(0), fpTrajectory(0),
     51     StoreTrajectory(0), verboseLevel(0), EventIsAborted(false)
     52 {
     53   fpSteppingManager = new G4SteppingManager();
     54   messenger = new G4TrackingMessenger(this);
     55 }

::

    067 void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
    ...
    069 {
    070 
    071   // Receiving a G4Track from the EventManager, this funciton has the
    072   // responsibility to trace the track till it stops.
    073   fpTrack = apValueG4Track;
    ...
    088   // Give SteppingManger the pointer to the track which will be tracked 
    089   fpSteppingManager->SetInitialStep(fpTrack);
    090 
    091   // Pre tracking user intervention process.
    092   fpTrajectory = 0;
    093   if( fpUserTrackingAction != 0 ) {
    094      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
    095   }
    ...
    109 
    110   // Give SteppingManger the maxmimum number of processes 
    111   fpSteppingManager->GetProcessNumber();
    112 
    113   // Give track the pointer to the Step
    114   fpTrack->SetStep(fpSteppingManager->GetStep());
    115 
    116   // Inform beginning of tracking to physics processes 
    117   fpTrack->GetDefinition()->GetProcessManager()->StartTracking(fpTrack);
    118 
    119   // Track the particle Step-by-Step while it is alive
    120   //  G4StepStatus stepStatus;
    121 
    122   while( (fpTrack->GetTrackStatus() == fAlive) ||
    123          (fpTrack->GetTrackStatus() == fStopButAlive) ){
    124 
    125     fpTrack->IncrementCurrentStepNumber();
    126     fpSteppingManager->Stepping();
    127 #ifdef G4_STORE_TRAJECTORY
    128     if(StoreTrajectory) fpTrajectory->
    129                         AppendStep(fpSteppingManager->GetStep());
    130 #endif
    131     if(EventIsAborted) {
    132       fpTrack->SetTrackStatus( fKillTrackAndSecondaries );
    133     }
    134   }
    135   // Inform end of tracking to physics processes 
    136   fpTrack->GetDefinition()->GetProcessManager()->EndTracking();
    137 
    138   // Post tracking user intervention process.
    139   if( fpUserTrackingAction != 0 ) {
    140      fpUserTrackingAction->PostUserTrackingAction(fpTrack);
    141   }
    142 
    143   // Destruct the trajectory if it was created
    144 #ifdef G4VERBOSE
    145   if(StoreTrajectory&&verboseLevel>10) fpTrajectory->ShowTrajectory();
    146 #endif
    147   if( (!StoreTrajectory)&&fpTrajectory ) {
    148       delete fpTrajectory;
    149       fpTrajectory = 0;
    150   }
    151 }



G4EventManager
----------------


`event/src/G4EventManager.cc`::

    099 void G4EventManager::DoProcessing(G4Event* anEvent)
    100 {
    145   sdManager = G4SDManager::GetSDMpointerIfExist();
    146   if(sdManager)
    147   { currentEvent->SetHCofThisEvent(sdManager->PrepareNewEvent()); }
    148 
    149   if(userEventAction) userEventAction->BeginOfEventAction(currentEvent);
    ...
    159   if(!abortRequested)
    160   { StackTracks( transformer->GimmePrimaries( currentEvent, trackIDCounter ),true ); }
    ...
    171   G4VTrajectory* previousTrajectory;
    172   while( ( track = trackContainer->PopNextTrack(&previousTrajectory) ) != 0 )
    173   {
    ...
    184     tracking = true;
    185     trackManager->ProcessOneTrack( track );
    186     istop = track->GetTrackStatus();
    187     tracking = false;
    ...
    217     G4TrackVector * secondaries = trackManager->GimmeSecondaries();
    218     switch (istop)
    219     {
    220       case fStopButAlive:
    221       case fSuspend:
    222         trackContainer->PushOneTrack( track, aTrajectory );
    223         StackTracks( secondaries );
    224         break;
    225 
    226       case fPostponeToNextEvent:
    227         trackContainer->PushOneTrack( track );
    228         StackTracks( secondaries );
    229         break;
    230 
    231       case fStopAndKill:
    232         StackTracks( secondaries );
    233         delete track;
    234         break;
    235 
    236       case fAlive:
    237         G4cout << "Illeagal TrackStatus returned from G4TrackingManager!"
    238              << G4endl;
    239       case fKillTrackAndSecondaries:
    240         //if( secondaries ) secondaries->clearAndDestroy();
    241         if( secondaries )
    242         {
    243           for(size_t i=0;i<secondaries->size();i++)
    244           { delete (*secondaries)[i]; }
    245           secondaries->clear();
    246         }
    247         delete track;
    248         break;
    249     }
    250   }
    ...
    260   if(sdManager)
    261   { sdManager->TerminateCurrentEvent(currentEvent->GetHCofThisEvent()); }
    262 
    263   if(userEventAction) userEventAction->EndOfEventAction(currentEvent);
    264 
    265   stateManager->SetNewState(G4State_GeomClosed);
    266   currentEvent = 0;
    267   abortRequested = false;
    268 }






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


#. can PostStepDoIt put it on hold ?


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




G4Transportation
------------------

Do nothing methods may be applicable `processes/transportation/include/G4Transportation.hh`::

    058 class G4Transportation : public G4VProcess
    059 {
    ...
    080      G4VParticleChange* PostStepDoIt(
    081                              const G4Track& track,
    082                              const G4Step&  stepData
    083                             );
    084        // Responsible for the relocation.
    085 
    086      G4double PostStepGetPhysicalInteractionLength(
    087                              const G4Track& ,
    088                              G4double   previousStepSize,
    089                              G4ForceCondition* pForceCond
    090                             );
    091        // Forces the PostStepDoIt action to be called, 
    092        // but does not limit the step.
    ...
    129      G4double AtRestGetPhysicalInteractionLength(
    130                              const G4Track& ,
    131                              G4ForceCondition*
    132                             ) { return -1.0; };
    133        // No operation in  AtRestDoIt.
    134 
    135      G4VParticleChange* AtRestDoIt(
    136                              const G4Track& ,
    137                              const G4Step&
    138                             ) {return 0;};
    139        // No operation in  AtRestDoIt.



NuWa hookup
-------------

#. Maybe split with `DetSimChroma/src/DsPhysConsChromaOptical.cc`

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsOptical.cc`::

    110 void DsPhysConsOptical::ConstructProcess()
    111 {
    169     G4OpAbsorption* absorb = 0;
    170     if (m_useAbsorption) {
    171         absorb = new G4OpAbsorption();
    172     }
    173 
    174     DsG4OpRayleigh* rayleigh = 0;
    175     if (m_useRayleigh) {
    176         rayleigh = new DsG4OpRayleigh();
    177     //        rayleigh->SetVerboseLevel(2);
    178     }
    179 
    180     //G4OpBoundaryProcess* boundproc = new G4OpBoundaryProcess();
    181     DsG4OpBoundaryProcess* boundproc = new DsG4OpBoundaryProcess();
    182     boundproc->SetModel(unified);
    183 
    184     G4FastSimulationManagerProcess* fast_sim_man
    185         = new G4FastSimulationManagerProcess("fast_sim_man");
    186 
    187     theParticleIterator->reset();
    188     while( (*theParticleIterator)() ) {
    189 
    190         G4ParticleDefinition* particle = theParticleIterator->value();
    191         G4ProcessManager* pmanager = particle->GetProcessManager();
    192 
    193         // Caution: as of G4.9, Cerenkov becomes a Discrete Process.
    194         // This code assumes a version of G4Cerenkov from before this version.
    195 
    196         if(cerenkov && cerenkov->IsApplicable(*particle)) {
    197             pmanager->AddProcess(cerenkov);
    198             pmanager->SetProcessOrdering(cerenkov, idxPostStep);
    199             debug() << "Process: adding Cherenkov to "
    200                     << particle->GetParticleName() << endreq;
    201         }
    202 
    203         if(scint && scint->IsApplicable(*particle)) {
    204             pmanager->AddProcess(scint);
    205             pmanager->SetProcessOrderingToLast(scint, idxAtRest);
    206             pmanager->SetProcessOrderingToLast(scint, idxPostStep);
    207             debug() << "Process: adding Scintillation to "
    208                     << particle->GetParticleName() << endreq;
    209         }
    210 
    211         if (particle == G4OpticalPhoton::Definition()) {
    212             if (absorb)
    213                 pmanager->AddDiscreteProcess(absorb);
    214             if (rayleigh)
    215                 pmanager->AddDiscreteProcess(rayleigh);
    216             pmanager->AddDiscreteProcess(boundproc);
    217             //pmanager->AddDiscreteProcess(pee);
    218             pmanager->AddDiscreteProcess(fast_sim_man);
    219         }
    220     }
    221 }



PhysicsList setup
~~~~~~~~~~~~~~~~~~~


* https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch06.html


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsGeneral.h`::

     12 class DsPhysConsGeneral : public GiGaPhysConstructorBase
     13 {
     14 public:
     15     DsPhysConsGeneral(const std::string& type,
     16                          const std::string& name,
     17                          const IInterface* parent );
     .. 
     20     // Interface methods
     21     void ConstructParticle();
     22     void ConstructProcess();


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsGeneral.cc`::

    098 void DsPhysConsGeneral::ConstructProcess()
    099 {
    100     // can't call this from a GiGaPhysConstructorBase, but
    101     // G4VModularPhysicsList will do it for us.
    102     // AddTransportation();


::

    [blyth@belle7 src]$ grep public\ GiGaPhysConstructorBase *.h
    DsPhysConsElectroNu.h:class DsPhysConsElectroNu : public GiGaPhysConstructorBase
    DsPhysConsEM.h:class DsPhysConsEM : public GiGaPhysConstructorBase
    DsPhysConsGeneral.h:class DsPhysConsGeneral : public GiGaPhysConstructorBase
    DsPhysConsHadron.h:class DsPhysConsHadron : public GiGaPhysConstructorBase
    DsPhysConsIon.h:class DsPhysConsIon : public GiGaPhysConstructorBase
    DsPhysConsOptical.h:class DsPhysConsOptical : public GiGaPhysConstructorBase




Impingement
~~~~~~~~~~~~

* http://www.dnp.fmph.uniba.sk/~zilka/work/g4course

* The G4ProcessManager has the possibility of switching on/off some processes
  at run time by using ActivateProcess() and InActivateProcess()

* The G4Transportation class must be registered with all particle
  classes. An AddTransportation() method is provided in
  G4VUserPhysicsList and it must be called in ConstructPhysics()







Wrapper Process
-----------------

Interesting for biasing, not canonically used  `processes/management/src/G4WrapperProcess.cc`::


     82 G4double G4WrapperProcess::
     83 PostStepGetPhysicalInteractionLength( const G4Track& track,
     84                                             G4double   previousStepSize,
     85                                             G4ForceCondition* condition )
     86 {
     87    return pRegProcess->PostStepGetPhysicalInteractionLength( track,
     88                                                              previousStepSize,
     89                                                              condition );
     90 }


G4WrapperProcess usage for event biasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://geant4.slac.stanford.edu/SLACTutorial14/EventBiasing.pdf

G4WrapperProcess can be used to implement user defined event biasing

#. G4WrapperProcess, which is a process itself, wraps an existing process

   * All function calls forwarded to wrapped process
   * Non-invasive way to modify behaviour of existing (built-in) process


1.  Create derived class inheriting from G4WrapperProcess
2.  Override only the methods to be modified, e.g., PostStepDoIt()
3.  Register this class in place of the original
4.  Finally, register the original (wrapped) process with user class
￼




