Geant4 StackManager
====================

Comments below aim at developing an external 
Optical Photon Propagation Stategy 



G4StackManager.hh
--------------------

#. the stacks are private, with no accessors
#. can setup multiple waiting stacks : that could be useful if need to multipart ZMQ message


`source/event/include/G4StackManager.hh`::

    049 // class description:
    050 //
    051 // This is the manager class of handling stacks of G4Track objects.
    052 // This class must be a singleton and be constructed by G4EventManager.
    053 // Almost all methods must be invoked exclusively by G4EventManager.
    054 // Especially, some Clear() methods MUST NOT be invoked by the user.
    055 // Event abortion is handled by G4EventManager.
    056 //
    057 // This G4StackingManager has three stacks, the urgent stack, the
    058 // waiting stack, and the postpone to next event stack. The meanings
    059 // of each stack is descrived in the Geant4 user's manual.
    060 //
    061 
    062 class G4StackManager
    063 {
    064   public:
    065       G4StackManager();
    066       ~G4StackManager();
    067 
    068   private:
    069       const G4StackManager& operator=(const G4StackManager &right);
    070       G4int operator==(const G4StackManager &right) const;
    071       G4int operator!=(const G4StackManager &right) const;
    072 
    073   public:
    074       G4int PushOneTrack(G4Track *newTrack, G4VTrajectory *newTrajectory = 0);
    075       G4Track * PopNextTrack(G4VTrajectory**newTrajectory);
    076       G4int PrepareNewEvent();
    077 
    078   public: // with description
    079       void ReClassify();
    080       //  Send all tracks stored in the Urgent stack one by one to 
    081       // the user's concrete ClassifyNewTrack() method. This method
    082       // can be invoked from the user's G4UserStackingAction concrete
    083       // class, especially fron its NewStage() method. Be aware that
    084       // when the urgent stack becomes empty, all tracks in the waiting
    085       // stack are send to the urgent stack and then the user's NewStage()
    086       // method is invoked.
    087 
    088       void SetNumberOfAdditionalWaitingStacks(G4int iAdd);
    089       //  Set the number of additional (optional) waiting stacks.
    090       // This method must be invoked at PreInit, Init or Idle states.
    091       // Once the user set the number of additional waiting stacks,
    092       // he/she can use the corresponding ENUM in G4ClassificationOfNewTrack.
    093       // The user should invoke G4RunManager::SetNumberOfAdditionalWaitingStacks
    094       // method, which invokes this method.
    095 
    096       void TransferStackedTracks(G4ClassificationOfNewTrack origin, G4ClassificationOfNewTrack destination);
    097       //  Transfter all stacked tracks from the origin stack to the destination stack.
    098       // The destination stack needs not be empty.
    099       // If the destination is fKill, tracks are deleted.
    100       // If the origin is fKill, nothing happen.
    101 
    102       void TransferOneStackedTrack(G4ClassificationOfNewTrack origin, G4ClassificationOfNewTrack destination);
    103       //  Transfter one stacked track from the origin stack to the destination stack.
    104       // The transfered track is the one which came last to the origin stack.
    105       // The destination stack needs not be empty.
    106       // If the destination is fKill, the track is deleted.
    107       // If the origin is fKill, nothing happen.
    108 
    109   private:
    110       G4UserStackingAction * userStackingAction;
    111       G4int verboseLevel;
    112 #ifdef G4_USESMARTSTACK
    113       G4SmartTrackStack * urgentStack;
    114 #else
    115       G4TrackStack * urgentStack;
    116 #endif
    117       G4TrackStack * waitingStack;
    118       G4TrackStack * postponeStack;
    119       G4StackingMessenger* theMessenger;
    120       std::vector<G4TrackStack*> additionalWaitingStacks;
    121       G4int numberOfAdditionalWaitingStacks;
    122 
    123   public:
    124       void clear();
    125       void ClearUrgentStack();
    126       void ClearWaitingStack(int i=0);
    127       void ClearPostponeStack();
    128       G4int GetNTotalTrack() const;
    129       G4int GetNUrgentTrack() const;
    130       G4int GetNWaitingTrack(int i=0) const;
    131       G4int GetNPostponedTrack() const;
    132       void SetVerboseLevel( G4int const value );
    133       void SetUserStackingAction(G4UserStackingAction* value);
    134 
    135   private:
    136      G4ClassificationOfNewTrack DefaultClassification(G4Track *aTrack);
    137 };
    138 
    139 #endif




G4StackManager::G4StackManager
-------------------------------

`source/event/src/G4StackManager.cc`::

    39 G4StackManager::G4StackManager()
    40 :userStackingAction(0),verboseLevel(0),numberOfAdditionalWaitingStacks(0)
    41 {
    42   theMessenger = new G4StackingMessenger(this);
    43 #ifdef G4_USESMARTSTACK
    44   urgentStack = new G4SmartTrackStack;
    45  // G4cout<<"+++ G4StackManager uses G4SmartTrackStack. +++"<<G4endl;
    46 #else
    47   urgentStack = new G4TrackStack(5000);
    48 //  G4cout<<"+++ G4StackManager uses ordinary G4TrackStack. +++"<<G4endl;
    49 #endif
    50   waitingStack = new G4TrackStack(1000);
    51   postponeStack = new G4TrackStack(1000);
    52 }
    ..


G4TrackStack
~~~~~~~~~~~~~

::

     45 // This is a stack class used by G4StackManager. This class object
     46 // stores G4StackedTrack class objects in the form of bi-directional
     47 // linked list.
     48 
     49 class G4TrackStack : public std::vector<G4StackedTrack>
     50 {
     51 public:
     52     G4TrackStack() : safetyValve1(0), safetyValve2(0), nstick(0) {}
     53   G4TrackStack(size_t n) : safetyValve1(4*n/5), safetyValve2(4*n/5-100), nstick(100) { reserve(n);}
     54   ~G4TrackStack();
     55 


G4StackedTrack
~~~~~~~~~~~~~~~~~

* **simple** holder for track and trajectory pointers 

  * this might provide an opportunity to avoid keeping huge wait stacks of photons 
    waiting around, whilst there Chroma copies are externally propagated 
  * maybe can get away with deleting the track (and not creating the trajectory) 
    and just keeping the G4StackedTrack alive as a placeholder 
  * the place holder can then be repopulated with tracks created from the 
    results of the external propagation 
     

`source/event/include/G4StackedTrack.hh`::

     41 //
     42 // This class is exclusively used by G4StackManager and G4TrackStack
     43 // classes for storing a G4Track object.
     44 
     45 class G4StackedTrack
     46 {
     47 public:
     48   G4StackedTrack() : track(0), trajectory(0) {}
     49   G4StackedTrack(G4Track* aTrack, G4VTrajectory* aTraj = 0) : track(aTrack), trajectory(aTraj) {}
     50   ~G4StackedTrack() {}
     51 
     52 private:
     53   G4Track* track;
     54   G4VTrajectory* trajectory;
     55 
     56 public:
     57   G4Track* GetTrack() const { return track; }
     58   G4VTrajectory* GetTrajectory() const { return trajectory; }
     59 };
     60 
     61 #endif


G4Track
~~~~~~~~~~

* its fat


`source/track/include/G4Track.hh`::

    072 //////////////
    073 class G4Track
    074 ////////////// 
    075 {
    076 
    077 //--------
    078 public: // With description
    079 
    080 // Constructor
    081    G4Track();
    082    G4Track(G4DynamicParticle* apValueDynamicParticle,
    083            G4double aValueTime,
    084            const G4ThreeVector& aValuePosition);
    085       // aValueTime is a global time
    086    G4Track(const G4Track&);
    087    // Copy Constructor copys members other than tracking information
    088 
    ...
    114    G4int GetTrackID() const;
    115    void SetTrackID(const G4int aValue);
    116 
    117    G4int GetParentID() const;
    118    void SetParentID(const G4int aValue);
    119 
    120   // dynamic particle 
    121    const G4DynamicParticle* GetDynamicParticle() const;



`source/track/src/G4Track.cc`::

    094 //////////////////
    095 G4Track::G4Track()
    096 //////////////////
    097   : fCurrentStepNumber(0),
    098     fGlobalTime(0),           fLocalTime(0.),
    099     fTrackLength(0.),
    100     fParentID(0),             fTrackID(0),
    101     fVelocity(c_light),
    102     fpDynamicParticle(0),
    103     fTrackStatus(fAlive),
    104     fBelowThreshold(false),   fGoodForTracking(false),
    105     fStepLength(0.0),         fWeight(1.0),
    106     fpStep(0),
    107     fVtxKineticEnergy(0.0),
    108     fpLVAtVertex(0),          fpCreatorProcess(0),
    109     fCreatorModelIndex(-1),
    110     fpUserInformation(0),
    111     prev_mat(0),  groupvel(0),
    112     prev_velocity(0.0), prev_momentum(0.0),
    113     is_OpticalPhoton(false),
    114     useGivenVelocity(false)
    115 {
    116 }

    /// default ctor : not so fat


G4DynamicParticle
~~~~~~~~~~~~~~~~~~
               
`source/particles/management/include/G4DynamicParticle.hh`::

     73 class G4DynamicParticle
     74 {
     75   // Class Description
     76   //  The dynamic particle is a class which contains the purely
     77   //  dynamic aspects of a moving particle. It also has a
     78   //  pointer to a G4ParticleDefinition object, which holds
     79   //  all the static information.
     80   //




G4StackManager::PushOneTrack
-------------------------------


::

    92 G4int G4StackManager::PushOneTrack(G4Track *newTrack,G4VTrajectory *newTrajectory)
    93 {
    ...
    166   G4ClassificationOfNewTrack classification = DefaultClassification( newTrack );
    167   if(userStackingAction)
    168   { classification = userStackingAction->ClassifyNewTrack( newTrack ); }

    ///     Maybe could in ClassifyNewTrack:
    ///
    ///          * collect OP trackinfo into ChromaPhotonList  
    ///          * delete the OP track, set pointer to NULL   (hmm its called `const G4Track*`, maybe need some patching)
    ///            [this would avoid pointlessly holding large memory expensive stacks of OPs ] 
    ///          * return fWaiting for OPs
    ///

    169 
    170   if(classification==fKill)   // delete newTrack without stacking
    171   {
    ...
    180     delete newTrack;
    181     delete newTrajectory;
    182   }
    183   else
    184   {
    185     G4StackedTrack newStackedTrack( newTrack, newTrajectory );
    186     switch (classification)
    187     {
    188       case fUrgent:
    189         urgentStack->PushToStack( newStackedTrack );
    190         break;
    191       case fWaiting:
    192         waitingStack->PushToStack( newStackedTrack );    
    ///                  newTrack could be NULL here without harm
    193         break;
    194       case fPostpone:
    195         postponeStack->PushToStack( newStackedTrack );
    196         break;
    197       default:
    198         G4int i = classification - 10;
    199         if(i<1||i>numberOfAdditionalWaitingStacks) {
    200           G4ExceptionDescription ED;
    201           ED << "invalid classification " << classification << G4endl;
    202           G4Exception("G4StackManager::PushOneTrack","Event0051",
    203           FatalException,ED);
    204         } else {
    205           additionalWaitingStacks[i-1]->PushToStack( newStackedTrack );
    206         }
    207         break;
    208     }
    209   }
    210 
    211   return GetNUrgentTrack();
    212 }


G4StackManager::PopNextTrack
-------------------------------

External Propagation Intervention Possibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At `NewStage`:

* have collected all the OPs
* have also seen all other tracks, so can judge whether the event is interesting to proceed with 

Thus within `NewStage` for interesting events:

* send ZMQ REQ message with serialized ChromaPhotonList  to Chroma server, 
* block until get REP, in the form of propagated ChromaPhotonList 

  * propagated till where ? OP subset ? enter PMT/hit Bialkali 
 
* then call `ReClassify` to revisit the track placeholders 

  * when receive NULL tracks, switch them for propagated tracks and classify `fUrgent`
  * for the extra NULLs (as only a subset of photons will come back) create a 
    sacrificial track and classify `fKill`

Maybe:

* do i need to match `trackID,parentID`  to keep G4 fooled ?  

  * probably not useful 
   
* how involved is creating tracks ?


Standard Operation
~~~~~~~~~~~~~~~~~~~~~~~

::

    084       // when the urgent stack becomes empty, all tracks in the waiting
    085       // stack are send to the urgent stack and then the user's NewStage()
    086       // method is invoked.
 

* pops from urgent until thats empty 
* then waiting stack transferred to urgent
* additional waiting stacks are shunted to one higher priority notch 
* `userStackingAction::NewStage` then gives opportunity to 

   * `ReClassify` OR `clear`
   * the `ReClassify` calls `ClassifyNewTrack` for each track that has been waiting 


::

    215 G4Track * G4StackManager::PopNextTrack(G4VTrajectory**newTrajectory)
    216 {
    225   while( GetNUrgentTrack() == 0 )
    226   {
    ...
    231     waitingStack->TransferTo(urgentStack);
    232     if(numberOfAdditionalWaitingStacks>0) {
    233       for(int i=0;i<numberOfAdditionalWaitingStacks;i++) {
    234         if(i==0) {
    235           additionalWaitingStacks[0]->TransferTo(waitingStack);
    236         } else {
    237           additionalWaitingStacks[i]->TransferTo(additionalWaitingStacks[i-1]);
    238         }
    239       }
    240     }
    241     if(userStackingAction) userStackingAction->NewStage();
    ...
    247     if( ( GetNUrgentTrack()==0 ) && ( GetNWaitingTrack()==0 ) ) return 0;
    248   }
    249 
    250   G4StackedTrack selectedStackedTrack = urgentStack->PopFromStack();
    251   G4Track * selectedTrack = selectedStackedTrack.GetTrack();
    252   *newTrajectory = selectedStackedTrack.GetTrajectory();
    ...
    265   return selectedTrack;
    266 }



G4StackManager::ReClassify
----------------------------

::


    268 void G4StackManager::ReClassify()
    269 {
    270   G4StackedTrack aStackedTrack;
    271   G4TrackStack tmpStack;
    272 
    273   if( !userStackingAction ) return;
    274   if( GetNUrgentTrack() == 0 ) return;
    275 
    276   urgentStack->TransferTo(&tmpStack);
    277   while( tmpStack.GetNTrack() > 0 )
    278   {
    279     aStackedTrack=tmpStack.PopFromStack();
    280     G4ClassificationOfNewTrack classification =
    281     userStackingAction->ClassifyNewTrack( aStackedTrack.GetTrack() );
    282     switch (classification)
    283     {
    284       case fKill:
    285         delete aStackedTrack.GetTrack();
    286         delete aStackedTrack.GetTrajectory();
    287         break;
    288       case fUrgent:
    289         urgentStack->PushToStack( aStackedTrack );
    290         break;
    291       case fWaiting:
    292         waitingStack->PushToStack( aStackedTrack );
    293         break;
    294       case fPostpone:
    295         postponeStack->PushToStack( aStackedTrack );
    296         break;
    297       default:
    298         G4int i = classification - 10;
    299         if(i<1||i>numberOfAdditionalWaitingStacks) {
    300           G4ExceptionDescription ED;
    301           ED << "invalid classification " << classification << G4endl;
    302           G4Exception("G4StackManager::ReClassify","Event0052",
    303                       FatalException,ED);
    304         } else {
    305           additionalWaitingStacks[i-1]->PushToStack( aStackedTrack );
    306         }
    307         break;
    308     }
    309   }
    310 }



PrepareNewEvent
----------------

::

    312 G4int G4StackManager::PrepareNewEvent()
    313 {
    314   if(userStackingAction) userStackingAction->PrepareNewEvent();
    315  
    316   urgentStack->clearAndDestroy(); // Set the urgentStack in a defined state. Not doing it would affect reproducibility.
    317  
    318   G4int n_passedFromPrevious = 0;
    319 
    320   if( GetNPostponedTrack() > 0 )
    321   {
    ...
    379   }
    380  
    381   return n_passedFromPrevious;
    382 }



