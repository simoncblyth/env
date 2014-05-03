Geant4 EventManager
====================


G4EventManager::DoProcessing
-------------------------------


* :doc:`geant4_stackmanager`


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
    ///   G4StackManager  userStackAction PrepareNewEvent called in turn
    /// 
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
    ///
    ///        StackTracks pushes each primary onto G4StackManager with  trackContainer->PushOneTrack 
    ///        at which point the user stacking action gets to Classify the track which directs which stack
    ///        the track is placed on  
    ///
    ...
    171   G4VTrajectory* previousTrajectory;
    172   while( ( track = trackContainer->PopNextTrack(&previousTrajectory) ) != 0 )
    173   {
    ...
    184     tracking = true;
    185     trackManager->ProcessOneTrack( track );
    ///     G4TrackingManager::ProcessOneTrack .... ALMOST 98% OF TIME IS SPENT IN THERE 
    ///
    186     istop = track->GetTrackStatus();
    187     tracking = false;
    ...
    198     G4VTrajectory * aTrajectory = 0;
    ...
    217     G4TrackVector * secondaries = trackManager->GimmeSecondaries();
    218     switch (istop)
    219     {
    220       case fStopButAlive:
    221       case fSuspend:
    222         trackContainer->PushOneTrack( track, aTrajectory );
    223         StackTracks( secondaries );
    ///         
    ///         pushing secondary tracks onto stack   
    ///
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



G4EventManager::StackTracks
------------------------------

::

    270 void G4EventManager::StackTracks(G4TrackVector *trackVector,G4bool IDhasAlreadySet)
    271 {
    272   G4Track * newTrack;
    273 
    274   if( trackVector )
    275   {
    276 
    277     size_t n_passedTrack = trackVector->size();
    278     if( n_passedTrack == 0 ) return;
    279     for( size_t i = 0; i < n_passedTrack; i++ )
    280     {
    281       newTrack = (*trackVector)[ i ];
    282       trackIDCounter++;
    283       if(!IDhasAlreadySet)
    284       {
    285         newTrack->SetTrackID( trackIDCounter );
    286         if(newTrack->GetDynamicParticle()->GetPrimaryParticle())
    287         {
    288           G4PrimaryParticle* pp
    289             = (G4PrimaryParticle*)(newTrack->GetDynamicParticle()->GetPrimaryParticle());
    290           pp->SetTrackID(trackIDCounter);
    291         }
    292       }
    293       newTrack->SetOriginTouchableHandle(newTrack->GetTouchableHandle());
    294       trackContainer->PushOneTrack( newTrack );
    ...
    304     }
    305     trackVector->clear();
    306   }
    307 }






`source/event/include/G4EventManager.hh`::

     54 // class description:
     55 //
     56 //  G4EventManager controls an event. This class must be a singleton
     57 //      and should be constructed by G4RunManager.
     58 //
     59 
     60 class G4EventManager
     61 {
     .. 
     76 
     77   public: // with description
     78       void ProcessOneEvent(G4Event* anEvent);
     79       //  This method is the main entry to this class for simulating an event.
     80 
     81       void ProcessOneEvent(G4TrackVector* trackVector,G4Event* anEvent=0);
     82       //  This is an alternative entry for large HEP experiments which create G4Track
     83       // objects by themselves directly without using G4VPrimaryGenerator or user
     84       // primary generator action. Dummy G4Event object will be created if "anEvent" is null
     85       // for internal use, but this dummy object will be deleted at the end of this
     86       // method and will never be available for the use after the processing.
     87       // Note that in this case of null G4Event pointer no output of the simulated event 
     88       // is returned by this method, but the user must implement some mechanism
     89       // of storing output by his/herself, e.g. in his/her UserEventAction and/or
     90       // sensitive detectors.
     91       //  If valid G4Event object is given, this object will not be deleted with
     92       // this method and output objects such as hits collections and trajectories
     93       // will be associated to this event object. If this event object has valid
     94       // primary vertices/particles, they will be added to the given trackvector input.
     95 
     96   private:
     97       void DoProcessing(G4Event* anEvent);
     98       void StackTracks(G4TrackVector *trackVector, G4bool IDhasAlreadySet=false);
     99 



