Geant4 Background
==================


G4UserTrackingAction
------------------------

* http://geant4.slac.stanford.edu/Tips/event/5.html

G4EventManager allows setting the G4UserTrackingAction on the G4TrackingManager::

    [blyth@cms01 source]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source
    [blyth@cms01 source]$ vi event/src/G4EventManager.cc

    308 void G4EventManager::SetUserAction(G4UserEventAction* userAction)
    309 {   
    310   userEventAction = userAction;
    311   if(userEventAction) userEventAction->SetEventManager(this);
    312 }
    313 
    314 void G4EventManager::SetUserAction(G4UserStackingAction* userAction)
    315 {
    316   userStackingAction = userAction;
    317   trackContainer->SetUserStackingAction(userAction);
    318 }
    319 
    320 void G4EventManager::SetUserAction(G4UserTrackingAction* userAction)
    321 {     
    322   userTrackingAction = userAction;
    323   trackManager->SetUserAction(userAction);
    324 }
    325 
    326 void G4EventManager::SetUserAction(G4UserSteppingAction* userAction)
    327 {
    328   userSteppingAction = userAction;
    329   trackManager->SetUserAction(userAction);
    330 }


The  `G4UserTrackingAction::PreUserTrackingAction` is invoked at `G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)`  
allowing track status changes, like kills::

     91 
     92   // Pre tracking user intervention process.
     93   fpTrajectory = 0;
     94   if( fpUserTrackingAction != NULL ) {
     95      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
     96   }

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H G4UserTrackingAction {} \;
    ./error_propagation/src/G4ErrorPropagator.cc:  const G4UserTrackingAction* fpUserTrackingAction =
    ./error_propagation/src/G4ErrorPropagator.cc:    const_cast<G4UserTrackingAction*>(fpUserTrackingAction)
    ./error_propagation/src/G4ErrorPropagator.cc:  const G4UserTrackingAction* fpUserTrackingAction =
    ./error_propagation/src/G4ErrorPropagator.cc:    const_cast<G4UserTrackingAction*>(fpUserTrackingAction)
    ./error_propagation/src/G4ErrorPropagatorManager.cc:void G4ErrorPropagatorManager::SetUserAction(G4UserTrackingAction* userAction)
    ./error_propagation/src/G4ErrorRunManagerHelper.cc:void G4ErrorRunManagerHelper::SetUserAction(G4UserTrackingAction* userAction)
    ./event/src/G4EventManager.cc:void G4EventManager::SetUserAction(G4UserTrackingAction* userAction)
    ./tracking/src/G4UserTrackingAction.cc:// $Id: G4UserTrackingAction.cc,v 1.10 2006/06/29 21:16:19 gunter Exp $
    ./tracking/src/G4UserTrackingAction.cc:// G4UserTrackingAction.cc
    ./tracking/src/G4UserTrackingAction.cc:#include "G4UserTrackingAction.hh"
    ./tracking/src/G4UserTrackingAction.cc:G4UserTrackingAction::G4UserTrackingAction()
    ./tracking/src/G4UserTrackingAction.cc:   msg =  " You are instantiating G4UserTrackingAction BEFORE your\n";
    ./tracking/src/G4UserTrackingAction.cc:   msg += "such as G4UserTrackingAction.";
    ./tracking/src/G4UserTrackingAction.cc:   G4Exception("G4UserTrackingAction::G4UserTrackingAction()",
    ./tracking/src/G4UserTrackingAction.cc:G4UserTrackingAction::~G4UserTrackingAction()
    ./tracking/src/G4UserTrackingAction.cc:void G4UserTrackingAction::
    [blyth@cms01 source]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source


::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H PreUserTrackingAction {} \;
    ./visualization/management/src/G4VisCommandsSceneAdd.cc:     "\nin PreUserTrackingAction.");
    ./visualization/RayTracer/src/G4RTTrackingAction.cc:void G4RTTrackingAction :: PreUserTrackingAction(const G4Track*)
    ./error_propagation/src/G4ErrorPropagator.cc:  InvokePreUserTrackingAction( theG4Track );  
    ./error_propagation/src/G4ErrorPropagator.cc:void G4ErrorPropagator::InvokePreUserTrackingAction( G4Track* fpTrack )
    ./error_propagation/src/G4ErrorPropagator.cc:      ->PreUserTrackingAction((fpTrack) );
    ./tracking/src/G4TrackingManager.cc:     fpUserTrackingAction->PreUserTrackingAction(fpTrack);


G4TrackStatus
----------------

::

    track/include/G4Track.hh

    174   // track status, flags for tracking
    175    G4TrackStatus GetTrackStatus() const;
    176    void SetTrackStatus(const G4TrackStatus aTrackStatus);


Curious, more states accessible cf StackAction classification::

     track/include/G4TrackStatus.hh

     49 //////////////////
     50 enum G4TrackStatus
     51 //////////////////
     52 {
     53 
     54   fAlive,             // Continue the tracking
     55   fStopButAlive,      // Invoke active rest physics processes and
     56                       // and kill the current track afterward
     57   fStopAndKill,       // Kill the current track
     58 
     59   fKillTrackAndSecondaries,
     60                       // Kill the current track and also associated
     61                       // secondaries.
     62   fSuspend,           // Suspend the current track
     63   fPostponeToNextEvent
     64                       // Postpones the tracking of thecurrent track 
     65                       // to the next event.
     66 
     67 };





