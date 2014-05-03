Geant4 TrackingAction
======================

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/UserActions/OptionalActions.html


Chroma PhotonTrackingAction::PreUserTrackingAction 
----------------------------------------------------

* collects OP tracks and sets their status to  `fStopAndKill`
* Can I somehow revive them later, after Chroma transport ?


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
    114     const_cast<G4Track *>(track)->SetTrackStatus(fStopAndKill);   // GETS RID OF CONST TO ALLOW STATUS CHANGE 
    115   }
    116 }

PreUserTrackingAction examples
--------------------------------

Casting away const in `PreUserTrackingAction` is standard practice in seems.::

    delta:geant4.10.00.p01 blyth$ find examples -name '*.cc' -exec grep -l PreUserTrackingAction {} \;
    examples/advanced/amsEcal/src/TrackingAction.cc
    examples/advanced/ChargeExchangeMC/src/CexmcTrackingAction.cc
    examples/advanced/medical_linac/src/ML2TrackingAction.cc
    examples/advanced/microdosimetry/src/TrackingAction.cc
    examples/advanced/microelectronics/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm1/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm10/src/Em10TrackingAction.cc
    examples/extended/electromagnetic/TestEm3/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm5/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm9/src/TrackingAction.cc
    examples/extended/eventgenerator/HepMC/HepMCEx01/src/ExN04TrackingAction.cc
    examples/extended/eventgenerator/HepMC/MCTruth/src/MCTruthTrackingAction.cc
    examples/extended/eventgenerator/particleGun/src/TrackingAction.cc
    examples/extended/field/field04/src/F04TrackingAction.cc
    examples/extended/hadronic/Hadr04/src/TrackingAction.cc
    examples/extended/medical/electronScattering/src/TrackingAction.cc
    examples/extended/medical/fanoCavity/src/TrackingAction.cc
    examples/extended/medical/fanoCavity2/src/TrackingAction.cc
    examples/extended/medical/GammaTherapy/src/TrackingAction.cc
    examples/extended/optical/LXe/src/LXeTrackingAction.cc
    examples/extended/optical/wls/src/WLSTrackingAction.cc
    examples/extended/parallel/TopC/ParN04/src/ExN04TrackingAction.cc
    examples/extended/radioactivedecay/rdecay01/src/TrackingAction.cc
    examples/extended/runAndEvent/RE01/src/RE01TrackingAction.cc
    examples/extended/runAndEvent/RE04/src/RE04TrackingAction.cc
    examples/extended/runAndEvent/RE05/src/RE05TrackingAction.cc


`examples/advanced/ChargeExchangeMC/src/CexmcTrackingAction.cc`::

    091 
    092     G4Track *  theTrack( const_cast< G4Track * >( track ) );
    093 
    ...
    154     if ( ! track->GetUserInformation() )
    155         theTrack->SetUserInformation( trackInfo );





G4UserTrackingAction
-----------------------


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



Examples
-----------

::

    delta:geant4.10.00.p01 blyth$ find examples -name '*.cc' -exec grep -l G4UserTrackingAction {} \;
    examples/advanced/amsEcal/src/TrackingAction.cc
    examples/advanced/ChargeExchangeMC/src/CexmcEventAction.cc
    examples/extended/electromagnetic/TestEm1/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm10/src/Em10TrackingAction.cc
    examples/extended/electromagnetic/TestEm11/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm12/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm2/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm3/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm5/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm7/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm9/src/TrackingAction.cc
    examples/extended/eventgenerator/HepMC/HepMCEx01/HepMCEx01.cc
    examples/extended/eventgenerator/particleGun/src/TrackingAction.cc
    examples/extended/exoticphysics/monopole/src/TrackingAction.cc
    examples/extended/field/BlineTracer/src/G4BlineTracer.cc
    examples/extended/hadronic/Hadr04/src/TrackingAction.cc
    examples/extended/parallel/TopC/ParN04/ParN04.cc
    examples/extended/radioactivedecay/rdecay01/src/TrackingAction.cc
    examples/extended/runAndEvent/RE01/src/RE01TrackingAction.cc
    examples/extended/runAndEvent/RE05/src/RE05ActionInitialization.cc
    delta:geant4.10.00.p01 blyth$ 


PostUserTrackingAction examples
--------------------------------

::

    delta:geant4.10.00.p01 blyth$ find examples -name '*.cc' -exec grep -l PostUserTrackingAction {} \;
    examples/advanced/amsEcal/src/TrackingAction.cc
    examples/advanced/purging_magnet/src/PurgMagTrackingAction.cc
    examples/extended/electromagnetic/TestEm1/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm11/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm12/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm2/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm3/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm5/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm7/src/TrackingAction.cc
    examples/extended/electromagnetic/TestEm9/src/TrackingAction.cc
    examples/extended/errorpropagation/errprop.cc
    examples/extended/eventgenerator/HepMC/MCTruth/src/MCTruthTrackingAction.cc
    examples/extended/eventgenerator/particleGun/src/TrackingAction.cc
    examples/extended/exoticphysics/monopole/src/TrackingAction.cc
    examples/extended/field/field04/src/F04TrackingAction.cc
    examples/extended/hadronic/Hadr04/src/TrackingAction.cc
    examples/extended/medical/electronScattering/src/TrackingAction.cc
    examples/extended/medical/fanoCavity/src/TrackingAction.cc
    examples/extended/medical/fanoCavity2/src/TrackingAction.cc
    examples/extended/medical/GammaTherapy/src/TrackingAction.cc
    examples/extended/optical/LXe/src/LXeTrackingAction.cc
    examples/extended/optical/wls/src/WLSTrackingAction.cc
    examples/extended/radioactivedecay/rdecay01/src/TrackingAction.cc
    examples/extended/runAndEvent/RE01/src/RE01TrackingAction.cc
    examples/extended/runAndEvent/RE04/src/RE04TrackingAction.cc
    delta:geant4.10.00.p01 blyth$ 



