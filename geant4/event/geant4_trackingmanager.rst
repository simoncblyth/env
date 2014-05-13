Geant4 TrackingManager
========================


G4TrackingManager.hh
----------------------

`source/tracking/include/G4TrackingManager.hh`::

    031 // G4TrackingManager.hh
    032 //
    033 // class description:
    034 //  This is an interface class among the event,  the track
    035 //  and the tracking category. It handles necessary 
    036 //  message passings between the upper hierarchical object, which 
    037 //  is the event manager (G4EventManager), and lower hierarchical 
    038 //  objects in the tracking category. It receives one track in an 
    039 //  event from the event manager and takes care to finish tracking it. 
    040 //  Geant4 kernel use only.
    ...
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


Chroma PreUserTrackingAction
------------------------------

* NB `const_cast<G4Track *>` gets rid of the **const** allowing the track status change 

`chroma/src/G4chroma.cc`::

    105 void PhotonTrackingAction::PreUserTrackingAction(const G4Track *track)
    106 {
    107   G4ParticleDefinition *particle = track->GetDefinition();
    108   if (particle->GetParticleName() == "opticalphoton") {
    109     pos.push_back(track->GetPosition()/mm);
    110     dir.push_back(track->GetMomentumDirection());
    111     pol.push_back(track->GetPolarization());
    112     wavelength.push_back( (h_Planck * c_light / track->GetKineticEnergy()) / nanometer );
    113     t0.push_back(track->GetGlobalTime() / ns);
    114     const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);
    115   }
    116 }


G4TrackingManager::ProcessOneTrack (98% OF TIME IN HERE)
-------------------------------------------------------------

`source/tracking/src/G4TrackingManager.cc`::


    066 ////////////////////////////////////////////////////////////////
    067 void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
    068 ////////////////////////////////////////////////////////////////
    069 {
    070 
    071   // Receiving a G4Track from the EventManager, this funciton has the
    072   // responsibility to trace the track till it stops.
    073   fpTrack = apValueG4Track;
    074   EventIsAborted = false;
    ...
    079   size_t itr;
    ...
    081   for(itr=0;itr<GimmeSecondaries()->size();itr++){
    082      delete (*GimmeSecondaries())[itr];
    083   }
    084   GimmeSecondaries()->clear();
    ...
    088   // Give SteppingManger the pointer to the track which will be tracked 
    089   fpSteppingManager->SetInitialStep(fpTrack);
    ///
    ///   G4SteppingManager
    ///
    090 
    091   // Pre tracking user intervention process.
    092   fpTrajectory = 0;
    093   if( fpUserTrackingAction != 0 ) {
    094      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
    095   }
    ///
    ///      Setting TrackStatus to smth other then fAlive, fStopButAlive   eg fStopAndKill
    ///      will skip the stepping : this will prevent stepping onto sensitive volumes
    ///      and hit formation for the track. And there is no way to intervene to revive it 
    ///      as PostUserTrackingAction really is *POST* 
    ///
    ///      Suggests better to intervene at stacking level, but can i diddle the tracks ?
    ///
    ...
    110   // Give SteppingManger the maxmimum number of processes 
    111   fpSteppingManager->GetProcessNumber();
    112 
    113   // Give track the pointer to the Step
    114   fpTrack->SetStep(fpSteppingManager->GetStep());
    115 
    116   // Inform beginning of tracking to physics processes 
    117   fpTrack->GetDefinition()->GetProcessManager()->StartTracking(fpTrack);
    118 
    ...
    ///     
    ///
    122   while( (fpTrack->GetTrackStatus() == fAlive) ||
    123          (fpTrack->GetTrackStatus() == fStopButAlive) ){
    124 
    125     fpTrack->IncrementCurrentStepNumber();
    126     fpSteppingManager->Stepping();
    ...
    131     if(EventIsAborted) {
    132       fpTrack->SetTrackStatus( fKillTrackAndSecondaries );
    133     }
    134   }
    ///        
    ///
    135   // Inform end of tracking to physics processes 
    136   fpTrack->GetDefinition()->GetProcessManager()->EndTracking();
    137 
    138   // Post tracking user intervention process.
    139   if( fpUserTrackingAction != 0 ) {
    140      fpUserTrackingAction->PostUserTrackingAction(fpTrack);
    ///         TO LATE TO DO ANYTHING OTHER THAN COLLECT INFORMATION ON THE TRACK        
    141   }
    ...
    151 }
    152 








