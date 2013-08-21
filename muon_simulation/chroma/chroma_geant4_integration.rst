Chroma Geant4 Interaction
==========================


Objective
----------

* harness massively parallel processing to propagate large numbers of optical photons




Open Questions
----------------

how/when to give OP tracks back to G4/reconstruction code ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe **DsOpGPUStackAction**

#. collect OP into `fWaiting` stack (similar to DsOpStackAction)  
#. at `NewStage` 

  * make **interesting-or-not judgement**
  * translate OP G4Tracks into numpy arrays ready for Chroma/GPU  
  * perform OP cohort external propagation on GPU

     * where to stop GPU propagation ? defined SD volume ?  

  * translate back from numpy arrays diddling the waiting G4Tracks [where/access?]

     * `NewStage` invokes a reclassify `stackManager->ReClassify();` giving access
        to all the tracks in the *ClassifyNewTrack* allowing diddling then like the photon reweighting of
        `G4ClassificationOfNewTrack DsFastMuonStackAction::ClassifyNewTrack (const G4Track* aTrack)`

  * resume the G4 tracks by returning as `fUrgent`, which should immediately proceed into sens det handling 

     * how does SD/hit handover work /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc


how does G4 OP propagation end ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


translation of DYB solid geomety into surface tris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



what about some  magic *optransport* physics process ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NO, as need to deal with the OP as a cohort, not individually.
Physics processes act on individual OP.


OP collection and propagation, kicked off where ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `G4ClassificationOfNewTrack DsOpStackAction::ClassifyNewTrack (const G4Track* aTrack)` 

   * assigns fWaiting status to OP, causing collection of OP tracks in the waiting stack 

* status is flipped to proceed with OP propagation only for events deemed to be interesting
* the judgement and kick-off happens in `void DsOpStackAction::NewStage()` which is invoked
  when the `fUrgent` stack is empty (ie everything other than the waiting tracks have been tracked) 
  and `fWaiting` stack has entries


* a similar structure seems good for GPU propagation

   * collect all the OP to benefit from massive parallelisation



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




Boost python C++ `_g4chroma`
-----------------------------

* `src/mute.cc` control G4 stdout
* `src/G4chroma.hh`
* `src/G4chroma.cc`

One-by-one collection and G4 `fStopAndKill` of optical photons.

Boost python module `_g4chroma` implementation in C++ providing a G4UserTrackingAction *PhotonTrackingAction* 
that collects opticalphotons and provides accessors to them, and snuffs them out with *fStopAndKill* ::

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


::

    simon:chroma blyth$ find . -name '*.*' -exec grep -H _g4chroma {} \;
    ./chroma/generator/g4gen.py:from chroma.generator import _g4chroma
    ./chroma/generator/g4gen.py:        self.physics_list = _g4chroma.ChromaPhysicsList()
    ./chroma/generator/g4gen.py:        self.tracking_action = _g4chroma.PhotonTrackingAction()
    ./setup.py:        Extension('chroma.generator._g4chroma',
    ./src/G4chroma.cc:BOOST_PYTHON_MODULE(_g4chroma)



geometry from CUDA photon propagation, in `photon.h`::

    584 __device__ int
    585 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    586                      bool use_weights=false)
    587 {
    588     Surface *surface = geometry->surfaces[s.surface_index];
    589 
    590     if (surface->model == SURFACE_COMPLEX)
    591         return propagate_complex(p, s, rng, surface, use_weights);
    592     else if (surface->model == SURFACE_WLS)
    593         return propagate_at_wls(p, s, rng, surface, use_weights);
    594     else {
    595         // use default surface model: do a combination of specular and
    596         // diffuse reflection, detection, and absorption based on relative
    597         // probabilties
    598 
    599         // since the surface properties are interpolated linearly, we are
    600         // guaranteed that they still sum to 1.0.
    601         float detect = interp_property(surface, p.wavelength, surface->detect);
    602         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    603         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    604         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    605 
    606         float uniform_sample = curand_uniform(&rng);



::

    simon:cuda blyth$ grep __shared__ *.*
    bvh.cu:    __shared__ unsigned long long min_area[128];
    bvh.cu:    __shared__ unsigned long long adjacent_area;
    daq.cu:    __shared__ int photon_id;
    daq.cu:    __shared__ int triangle_id;
    daq.cu:    __shared__ int solid_id;
    daq.cu:    __shared__ int channel_index;
    daq.cu:    __shared__ unsigned int history;
    daq.cu:    __shared__ float photon_time;
    daq.cu:    __shared__ float weight;
    mesh.h:    __shared__ Geometry sg;
    pdf.cu:    __shared__ float distance_table[1000];
    pdf.cu:    __shared__ unsigned int *work_queue;
    pdf.cu:    __shared__ int queue_items;
    pdf.cu:    __shared__ int channel_id;
    pdf.cu:    __shared__ float channel_event_time;
    pdf.cu:    __shared__ int distance_table_len;
    pdf.cu:    __shared__ int offset;
    propagate.cu:    __shared__ unsigned int counter;
    propagate.cu:    __shared__ Geometry sg;
    render.cu:    __shared__ Geometry sg;
    simon:cuda blyth$ 


::

    simon:cuda blyth$ grep sync *.*      
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    bvh.cu:    __syncthreads();
    daq.cu:    __syncthreads();
    mesh.h:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    pdf.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    propagate.cu:    __syncthreads();
    render.cu:    __syncthreads();
    simon:cuda blyth$ 



