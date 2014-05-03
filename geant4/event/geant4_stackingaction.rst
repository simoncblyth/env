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




