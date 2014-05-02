Geant4 StackingAction
==========================

Documentation
---------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/UserActions/OptionalActions.html


G4UserStackingAction
---------------------


`source/event/include/G4UserStackingAction.hh`::

     44 class G4UserStackingAction
     45 {
     46   public:
     47       G4UserStackingAction();
     48       virtual ~G4UserStackingAction();
     49   protected:
     50       G4StackManager * stackManager;
     51   public:
     52       inline void SetStackManager(G4StackManager * value)
     53       { stackManager = value; }
     54 
     55   public: // with description
     56 //---------------------------------------------------------------
     57 // vitual methods to be implemented by user
     58 //---------------------------------------------------------------
     59 //
     60       virtual G4ClassificationOfNewTrack
     61         ClassifyNewTrack(const G4Track* aTrack);



StackingAction Source 
----------------------

::

    delta:geant4.10.00.p01 blyth$ find source -name '*.cc' -exec grep -l StackingAction {} \;

    source/run/src/G4RunManager.cc                         #  just holds userStackingAction property 
    source/event/src/G4EventManager.cc                     #  just holds userStackingAction property also
    source/event/src/G4StackManager.cc                     #  invokes the action and acts on classification in G4StackManager::PushOneTrack

    source/event/src/G4AdjointStackingAction.cc
    source/event/src/G4UserStackingAction.cc               # default empty-ish implementation

    source/run/src/G4AdjointSimManager.cc
    source/run/src/G4MaterialScanner.cc
    source/run/src/G4MTRunManager.cc
    source/run/src/G4VUserActionInitialization.cc
    source/run/src/G4WorkerRunManager.cc

    source/visualization/RayTracer/src/G4RTWorkerInitialization.cc
    source/visualization/RayTracer/src/G4TheRayTracer.cc


::

    813 void G4RunManager::SetUserAction(G4UserStackingAction* userAction)
    814 {
    815   eventManager->SetUserAction(userAction);
    816   userStackingAction = userAction;
    817 }

    315 void G4EventManager::SetUserAction(G4UserStackingAction* userAction)
    316 {
    317   userStackingAction = userAction;
    318   trackContainer->SetUserStackingAction(userAction);        ## trackContainer is G4StackManager
    319 }




::

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
    92 G4int G4StackManager::PushOneTrack(G4Track *newTrack,G4VTrajectory *newTrajectory)
    93 {
    ...
    166   G4ClassificationOfNewTrack classification = DefaultClassification( newTrack );
    167   if(userStackingAction)
    168   { classification = userStackingAction->ClassifyNewTrack( newTrack ); }
    169 
    170   if(classification==fKill)   // delete newTrack without stacking
    171   {
    172 #ifdef G4VERBOSE
    173     if( verboseLevel > 1 )
    174     {
    175       G4cout << "   ---> G4Track " << newTrack << " (trackID "
    176      << newTrack->GetTrackID() << ", parentID "
    177      << newTrack->GetParentID() << ") is not to be stored." << G4endl;
    178     }
    179 #endif
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



::

     57 G4ClassificationOfNewTrack G4UserStackingAction::ClassifyNewTrack
     58 (const G4Track*)
     59 {
     60   return fUrgent;
     61 }
     62 
     63 void G4UserStackingAction::NewStage()
     64 {;}
     65 
     66 void G4UserStackingAction::PrepareNewEvent()
     67 {;}
     68 




