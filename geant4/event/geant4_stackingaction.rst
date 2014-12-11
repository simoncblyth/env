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



Documentation Extracts
-----------------------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/UserActions/OptionalActions.html

ClassifyNewTrack
~~~~~~~~~~~~~~~~~

`ClassifyNewTrack()` is invoked by `G4StackManager` whenever a new `G4Track`
object is "pushed" onto a stack by `G4EventManager`.  `ClassifyNewTrack()`
returns an enumerator, `G4ClassificationOfNewTrack`, whose value indicates to
which stack, if any, the track will be sent.  This value should be determined
by the user. `G4ClassificationOfNewTrack` has four possible values:

#. `fUrgent` - track is placed in the urgent stack
#. `fWaiting` - track is placed in the waiting stack, and will not be simulated until the urgent stack is empty
#. `fPostpone` - track is postponed to the next event
#. `fKill` - the track is deleted immediately and not stored in any stack.

These assignments may be made based on the origin of the track which is obtained as follows:
`G4int parent_ID = aTrack->get_parentID();`
where

#. `parent_ID = 0` indicates a primary particle
#. `parent_ID > 0` indicates a secondary particle
#. `parent_ID < 0` indicates postponed particle from previous event.


* always returning `fUrgent` would result in stepping proceeding 
  for tracks as they arise, ie with OP mixed up with everything else


NewStage : deal with the wait stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`NewStage()` is invoked when the urgent stack is empty and the waiting stack
contains at least one `G4Track` object.  Here the user may kill or re-assign to
different stacks all the tracks in the waiting stack by calling the
`stackManager->ReClassify()` method which, in turn, calls the
`ClassifyNewTrack()` method. 

If no user action is taken, all tracks in the waiting stack are transferred to
the urgent stack. 

* HMM: so can see how long the Optical Photons are taking could first 
  collect them into the wait stack and then "do nothing" at NewStage


The user may also decide to abort the current event even though some tracks may
remain in the waiting stack by calling `stackManager->clear()`.  This method is
valid and safe only if it is called from the `G4UserStackingAction` class. 

A global method of event abortion is::

    G4UImanager * UImanager = G4UImanager::GetUIpointer();
    UImanager->ApplyCommand("/event/abort");


PrepareNewEvent
~~~~~~~~~~~~~~~~~~~

`PrepareNewEvent()` is invoked at the beginning of each event. At this point no
primary particles have been converted to tracks, so the urgent and waiting
stacks are empty. However, there may be tracks in the postponed-to-next-event
stack; for each of these the `ClassifyNewTrack()` method is called and the track
is assigned to the appropriate stack.





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




StackingAction examples
-------------------------

::

    delta:geant4.10.00.p01 blyth$ find examples -name '*.cc' -exec grep -l ClassifyNewTrack {} \;
    examples/advanced/composite_calorimeter/src/CCalStackingAction.cc
    examples/advanced/underground_physics/src/DMXStackingAction.cc
    examples/basic/B3/src/B3StackingAction.cc
    examples/extended/electromagnetic/TestEm1/src/StackingAction.cc
    examples/extended/electromagnetic/TestEm17/src/StackingAction.cc
    examples/extended/electromagnetic/TestEm18/src/StackingAction.cc
    examples/extended/electromagnetic/TestEm5/src/StackingAction.cc
    examples/extended/electromagnetic/TestEm6/src/StackingAction.cc
    examples/extended/electromagnetic/TestEm8/src/StackingAction.cc
    examples/extended/eventgenerator/HepMC/HepMCEx01/src/ExN04StackingAction.cc
    examples/extended/exoticphysics/phonon/src/XPhononStackingAction.cc
    examples/extended/exoticphysics/phonon/src/XPrimaryGeneratorAction.cc
    examples/extended/field/field04/src/F04StackingAction.cc
    examples/extended/hadronic/Hadr01/src/StackingAction.cc
    examples/extended/hadronic/Hadr02/src/StackingAction.cc
    examples/extended/hadronic/Hadr04/src/StackingAction.cc
    examples/extended/medical/fanoCavity/src/StackingAction.cc
    examples/extended/optical/LXe/src/LXeStackingAction.cc
    examples/extended/optical/OpNovice/src/OpNoviceStackingAction.cc
    examples/extended/optical/wls/src/WLSStackingAction.cc
    examples/extended/parallel/TopC/ParN04/src/ExN04StackingAction.cc
    examples/extended/runAndEvent/RE01/src/RE01StackingAction.cc
    examples/extended/runAndEvent/RE05/src/RE05StackingAction.cc
    delta:geant4.10.00.p01 blyth$ 




