Geant4 TrackingAction
======================

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/UserActions/OptionalActions.html



`source/tracking/include/G4UserTrackingAction.hh`::

     32 // G4UserTrackingAction.hh
     33 //
     34 // class description:
     35 //   This class represents actions taken place by the user at 
     36 //   the start/end point of processing one track. 
     ..
     53 ///////////////////////////
     54 class G4UserTrackingAction
     55 ///////////////////////////
     56 {
     57 
     58 //--------
     59 public: // with description
     60 //--------
     61 
     62 // Constructor & Destructor
     63    G4UserTrackingAction();
     64    virtual ~G4UserTrackingAction();
     65 
     66 // Member functions
     67    void SetTrackingManagerPointer(G4TrackingManager* pValue);
     68    virtual void PreUserTrackingAction(const G4Track*){;}
     69    virtual void PostUserTrackingAction(const G4Track*){;}
     70 
     71 //----------- 
     72    protected:
     73 //----------- 
     74 
     75 // Member data
     76    G4TrackingManager* fpTrackingManager;
     77 
     78 };
     79 


::

    321 void G4EventManager::SetUserAction(G4UserTrackingAction* userAction)
    322 {
    323   userTrackingAction = userAction;
    324   trackManager->SetUserAction(userAction);
    325 }





::

    delta:geant4.10.00.p01 blyth$ find source -name '*.cc' -exec grep -l G4UserTrackingAction {} \;
    source/error_propagation/src/G4ErrorPropagator.cc
    source/error_propagation/src/G4ErrorPropagatorManager.cc
    source/error_propagation/src/G4ErrorRunManagerHelper.cc
    source/event/src/G4EventManager.cc             ## holds pointer, but just passes along to G4TrackingManager
    source/run/src/G4AdjointSimManager.cc
    source/run/src/G4MTRunManager.cc
    source/run/src/G4RunManager.cc
    source/run/src/G4VUserActionInitialization.cc
    source/run/src/G4WorkerRunManager.cc
    source/tracking/src/G4UserTrackingAction.cc
    source/visualization/RayTracer/src/G4RTWorkerInitialization.cc


    delta:geant4.10.00.p01 blyth$ find source -name '*.hh' -exec grep -l G4UserTrackingAction {} \;
    source/error_propagation/include/G4ErrorPropagator.hh
    source/error_propagation/include/G4ErrorPropagatorManager.hh
    source/error_propagation/include/G4ErrorRunManagerHelper.hh
    source/event/include/G4EventManager.hh
    source/processes/electromagnetic/dna/management/include/G4ITStepProcessor.hh
    source/processes/electromagnetic/dna/management/include/G4ITTrackingInteractivity.hh
    source/processes/electromagnetic/dna/management/include/G4ITTrackingManager.hh
    source/run/include/G4AdjointSimManager.hh
    source/run/include/G4MaterialScanner.hh
    source/run/include/G4MTRunManager.hh
    source/run/include/G4RunManager.hh
    source/run/include/G4UserWorkerThreadInitialization.hh
    source/run/include/G4VUserActionInitialization.hh
    source/run/include/G4WorkerRunManager.hh
    source/track/include/G4VUserTrackInformation.hh
    source/tracking/include/G4AdjointTrackingAction.hh
    source/tracking/include/G4TrackingManager.hh
    source/tracking/include/G4UserTrackingAction.hh
    source/visualization/RayTracer/include/G4RTTrackingAction.hh
    source/visualization/RayTracer/include/G4RTWorkerInitialization.hh
    source/visualization/RayTracer/include/G4TheRayTracer.hh





G4EventManager::DoProcessing
-------------------------------

::

    099 void G4EventManager::DoProcessing(G4Event* anEvent)
    100 {
    ...
    122   G4ThreeVector center(0,0,0);
    123   G4Navigator* navigator =
    124       G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    125   navigator->LocateGlobalPointAndSetup(center,0,false);
    126 
    127   G4Track * track;
    128   G4TrackStatus istop;
    ...
    139   trackContainer->PrepareNewEvent();
    ///   G4StackManager  
    ...
    145   sdManager = G4SDManager::GetSDMpointerIfExist();
    ///   G4SDManager 
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
    ///     G4TrackingManager
    186     istop = track->GetTrackStatus();
    187     tracking = false;
    ...
    198     G4VTrajectory * aTrajectory = 0;
    ...
    216 
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



G4TrackingManager
-------------------

`source/tracking/include/G4TrackingManager.hh`::

    113 // Other member functions
    114 
    115    void ProcessOneTrack(G4Track* apValueG4Track);
    116       // Invoking this function, a G4Track given by the argument
    117       // will be tracked.  
    118 
    119    void EventAborted();
    120       // Invoking this function, the current tracking will be
    121       // aborted immediately. The tracking will return the 
    122       // G4TrackStatus in 'fUserKillTrackAndSecondaries'.
    123       // By this the EventManager deletes the current track and all 
    124       // its accoicated csecondaries.


::

    066 ////////////////////////////////////////////////////////////////
    067 void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
    068 ////////////////////////////////////////////////////////////////
    069 {
    070 
    071   // Receiving a G4Track from the EventManager, this funciton has the
    072   // responsibility to trace the track till it stops.
    073   fpTrack = apValueG4Track;
    074   EventIsAborted = false;
    075 
    076   // Clear 2ndary particle vector
    077   //  GimmeSecondaries()->clearAndDestroy();    
    078   //  std::vector<G4Track*>::iterator itr;
    079   size_t itr;
    080   //  for(itr=GimmeSecondaries()->begin();itr=GimmeSecondaries()->end();itr++){ 
    081   for(itr=0;itr<GimmeSecondaries()->size();itr++){
    082      delete (*GimmeSecondaries())[itr];
    083   }
    084   GimmeSecondaries()->clear();
    085 
    086   if(verboseLevel>0 && (G4VSteppingVerbose::GetSilent()!=1) ) TrackBanner();
    087 
    088   // Give SteppingManger the pointer to the track which will be tracked 
    089   fpSteppingManager->SetInitialStep(fpTrack);
    090 
    091   // Pre tracking user intervention process.
    092   fpTrajectory = 0;
    093   if( fpUserTrackingAction != 0 ) {
    094      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
    095   }
    096 #ifdef G4_STORE_TRAJECTORY
    097   // Construct a trajectory if it is requested
    098   if(StoreTrajectory&&(!fpTrajectory)) {
    099     // default trajectory concrete class object
    100     switch (StoreTrajectory) {
    101     default:
    102     case 1: fpTrajectory = new G4Trajectory(fpTrack); break;
    103     case 2: fpTrajectory = new G4SmoothTrajectory(fpTrack); break;
    104     case 3: fpTrajectory = new G4RichTrajectory(fpTrack); break;
    105     case 4: fpTrajectory = new G4RichTrajectory(fpTrack); break;
    106     }
    107   }
    108 #endif
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
    152 






