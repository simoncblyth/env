Geant4 API
===========

Review `G4UserXXXAction` API with eye to external optical photon 
propagation (a la Chroma).
Chroma grabs and kills optical photons tracks 
in `G4UserTrackingAction::PreUserTrackingAction` and seemingly doesnt give them back.


G4EventManager Overview
------------------------

* http://www.tunl.duke.edu/~tajima/geant4/pdf/jun2009b/1/kernel.pdf

* An event is the basic unit of simulation in Geant4. 
* At beginning of processing, primary tracks are generated. These primary tracks are pushed into a stack. 
* A track is popped up from the stack one by one and tracked. Resulting secondary tracks are pushed into the stack. 
 
    * This tracking lasts as long as the stack has a track. 
    * When the stack becomes empty, processing of one event is over. 



Optional User Actions
-------------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/UserActions/OptionalActions.html



G4EventManager::DoProcessing
------------------------------

::

    098 void G4EventManager::DoProcessing(G4Event* anEvent)
    ...
    139   trackContainer->PrepareNewEvent();
    ...
    145   sdManager = G4SDManager::GetSDMpointerIfExist();
    146   if(sdManager)
    147   { currentEvent->SetHCofThisEvent(sdManager->PrepareNewEvent()); }
    148 
    149   if(userEventAction) userEventAction->BeginOfEventAction(currentEvent);
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





G4UserEventAction 
-------------------

G4UserStackingAction
---------------------

`ClassifyNewTrack()` 
~~~~~~~~~~~~~~~~~~~~~~
      
* invoked by G4StackManager whenever a new G4Track object is "pushed" onto a stack by G4EventManager
* returns an enumerator, G4ClassificationOfNewTrack with one of four possible values:

    * fUrgent - track is placed in the urgent stack
    * fWaiting - track is placed in the waiting stack, and will not be simulated until the urgent stack is empty
    * fPostpone - track is postponed to the next event
    * fKill - the track is deleted immediately and not stored in any stack.

`NewStage()`
~~~~~~~~~~~~~

Invoked when the urgent stack is empty and the waiting stack contains at least one G4Track object. 
Here the user may kill or re-assign to different stacks all the tracks in the
waiting stack by calling the `stackManager->ReClassify()` method which, in turn,
calls the `ClassifyNewTrack()` method. 
If no user action is taken, all tracks in the waiting stack are transferred to the urgent stack. 
The user may also decide to abort the current event even though some tracks may remain in the waiting
stack by calling `stackManager->clear()`. 
This method is valid and safe only if it is called from the G4UserStackingAction class. 

`PrepareNewEvent()`
~~~~~~~~~~~~~~~~~~~~~

Invoked at the beginning of each event. At this point no primary particles
have been converted to tracks, so the urgent and waiting stacks are empty.
However, there may be tracks in the postponed-to-next-event stack; for each of
these the `ClassifyNewTrack()` method is called and the track is assigned to the
appropriate stack.


Examples DsFastMuonStackAction, DsOpStackAction, ExampleN04
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 src]$ vi /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsFastMuonStackAction.cc

* http://dayabay.bnl.gov/dox/DetSim/html/DsOpStackAction_8cc_source.html






G4UserTrackingAction
---------------------


G4UserSteppingAction
---------------------



G4SDManager
------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04s04.html
* http://www.ge.infn.it/geant4/training/portland/readout.pdf


ProcessHits
~~~~~~~~~~~~

::

    [blyth@cms01 src]$ grep ProcessHits *.cc
    DsPmtSensDet.cc:bool DsPmtSensDet::ProcessHits(G4Step* step,
    DsPmtSensDet.cc:        error() << "ProcessHits: step has no or empty touchable history" << endreq;
    DsRpcSensDet.cc:bool DsRpcSensDet::ProcessHits(G4Step* step,
    DsRpcSensDet.cc:      error() << "ProcessHits: step has no or empty touchable history." 
    [blyth@cms01 src]$ 



Registering hits on sensitive detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe possible lookahead to the next step with `fStep->GetPostStepPoint()->GetSensitiveDetector()`
to see if about to hit.

Whats the Chroma equivalent of SD "Collision detection"

      
::

        42     42  117: G4StepStatus G4SteppingManager::Stepping()
         .      .  118: //////////////////////////////////////////
         .      .  119: {
         .      .  120: 
         .      .  121: //--------
         .      .  122: // Prelude
         .      .  123: //--------
         .      .  124: #ifdef G4VERBOSE
         .      .  125:             // !!!!! Verbose
        10     10  126:              if(verboseLevel>0) fVerbose->NewStep();
         .      .  127:          else 
        13     13  128:              if(verboseLevel==-1) { 
         .      .  129:              G4VSteppingVerbose::SetSilent(1);
         .      .  130:          }
         .      .  131:          else
         9    246  132:              G4VSteppingVerbose::SetSilent(0);
         .      .  133: #endif 
         .      .  134: 
         .      .  135: // Store last PostStepPoint to PreStepPoint, and swap current and nex
         .      .  136: // volume information of G4Track. Reset total energy deposit in one Step. 
       169   1759  137:    fStep->CopyPostToPreStepPoint();
       265    317  138:    fStep->ResetTotalEnergyDeposit();
         .      .  139: 
         .      .  140: // Switch next touchable in track to current one
       390   9025  141:    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
                   ...
                   ... 
                   ... 
         .      .  218: //-------
         .      .  219: // Finale
         .      .  220: //-------
         .      .  221: 
         .      .  222: // Update 'TrackLength' and remeber the Step length of the current Step
        69    144  223:    fTrack->AddTrackLength(fStep->GetStepLength());
        85     91  224:    fPreviousStepSize = fStep->GetStepLength();
        21     36  225:    fStep->SetTrack(fTrack);
         .      .  226: #ifdef G4VERBOSE
         .      .  227:                          // !!!!! Verbose
         .      .  228: 
       108    108  229:            if(verboseLevel>0) fVerbose->StepInfo();
         .      .  230: #endif
         .      .  231: // Send G4Step information to Hit/Dig if the volume is sensitive
       270   2794  232:    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
       288    296  233:    StepControlFlag =  fStep->GetControlFlag();
        29     29  234:    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
         .      .  235:       fSensitive = fStep->GetPreStepPoint()->
       258    278  236:                                    GetSensitiveDetector();
        26     26  237:       if( fSensitive != 0 ) {
         1   4325  238:         fSensitive->Hit(fStep);
         .      .  239:       }
         .      .  240:    }
 



